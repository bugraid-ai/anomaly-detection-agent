"""
Anomaly Detection FastAPI Service with LangGraph Integration

This service:
1. Connects to Valkey to retrieve incident logs
2. Processes logs for anomaly detection using ML and statistical methods
3. Returns detected anomalies for suspect generation
4. Uses LangGraph for workflow orchestration
"""

import json
import logging
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import redis.asyncio as redis
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import Annotated, TypedDict

# Configuration
VALKEY_USERNAME = os.getenv("VALKEY_USERNAME", "chunking-lambda-valkey-read")
VALKEY_PASSWORD = os.getenv("VALKEY_PASSWORD", "ICOn1qVwMVA9GIb7")
VALKEY_HOST = os.getenv("VALKEY_HOST", "valkey-store-dev-zizz73.serverless.apse1.cache.amazonaws.com")
VALKEY_PORT = int(os.getenv("VALKEY_PORT", "6379"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyRequest(BaseModel):
    incident_id: str = Field(..., description="Incident ID (e.g., bugraid-INC-XXX)")
    time_window_hours: Optional[int] = Field(default=24, description="Time window for log analysis")
    confidence_threshold: Optional[float] = Field(default=0.7, description="Minimum confidence for anomaly detection")

class AnomalyResponse(BaseModel):
    incident_id: str
    anomalies_detected: int
    anomalies: List[Dict[str, Any]]
    processed_logs: int
    analysis_timestamp: str

class WorkflowState(TypedDict):
    """State for the anomaly detection workflow"""
    incident_id: str
    time_window_hours: int
    confidence_threshold: float
    logs: List[Dict[str, Any]]
    processed_logs: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    messages: Annotated[List[AnyMessage], add_messages]

class AnomalyDetectionService:
    def __init__(self):
        self.redis_client = None
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
        
    async def initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=VALKEY_HOST,
                port=VALKEY_PORT,
                username=VALKEY_USERNAME,
                password=VALKEY_PASSWORD,
                decode_responses=True,
                socket_timeout=10.0,
                socket_connect_timeout=10.0
            )
            await self.redis_client.ping()
            logger.info(f"Connected to Valkey at {VALKEY_HOST}:{VALKEY_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Valkey: {e}")
            raise
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for anomaly detection"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("fetch_logs", self.fetch_logs_node)
        workflow.add_node("preprocess_logs", self.preprocess_logs_node)
        workflow.add_node("detect_anomalies", self.detect_anomalies_node)
        workflow.add_node("format_results", self.format_results_node)
        
        # Add edges
        workflow.add_edge(START, "fetch_logs")
        workflow.add_edge("fetch_logs", "preprocess_logs")
        workflow.add_edge("preprocess_logs", "detect_anomalies")
        workflow.add_edge("detect_anomalies", "format_results")
        workflow.add_edge("format_results", END)
        
        return workflow.compile()
    
    async def fetch_logs_node(self, state: WorkflowState) -> WorkflowState:
        """Fetch logs from Valkey for the given incident"""
        logger.info(f"Fetching logs for incident {state['incident_id']}")
        
        try:
            # Get incident logs from Valkey
            incident_key = f"incident:{state['incident_id']}:logs"
            log_keys = await self.redis_client.lrange(incident_key, 0, -1)
            
            logs = []
            for log_key in log_keys:
                log_data = await self.redis_client.get(log_key)
                if log_data:
                    try:
                        log_entry = json.loads(log_data)
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse log entry: {log_key}")
            
            logger.info(f"Fetched {len(logs)} logs for incident {state['incident_id']}")
            state["logs"] = logs
            
        except Exception as e:
            logger.error(f"Error fetching logs: {e}")
            state["logs"] = []
        
        return state
    
    async def preprocess_logs_node(self, state: WorkflowState) -> WorkflowState:
        """Preprocess logs for anomaly detection"""
        logger.info(f"Preprocessing {len(state['logs'])} logs")
        
        processed_logs = []
        
        for log in state["logs"]:
            try:
                # Extract numerical features from logs
                processed_log = {
                    "timestamp": log.get("timestamp", ""),
                    "level": log.get("level", "INFO"),
                    "message": log.get("message", ""),
                    "service": log.get("service", ""),
                    "features": self._extract_log_features(log)
                }
                processed_logs.append(processed_log)
            except Exception as e:
                logger.warning(f"Error processing log: {e}")
        
        state["processed_logs"] = processed_logs
        logger.info(f"Preprocessed {len(processed_logs)} logs")
        
        return state
    
    def _extract_log_features(self, log: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from a log entry"""
        features = {}
        
        # Message length
        message = log.get("message", "")
        features["message_length"] = len(message)
        
        # Error keywords count
        error_keywords = ["error", "exception", "failed", "timeout", "crash"]
        features["error_keywords"] = sum(1 for keyword in error_keywords if keyword.lower() in message.lower())
        
        # Log level severity (numeric)
        level_severity = {
            "DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4, "CRITICAL": 5
        }
        features["level_severity"] = level_severity.get(log.get("level", "INFO"), 2)
        
        # Response time (if available)
        if "response_time" in log:
            try:
                features["response_time"] = float(log["response_time"])
            except (ValueError, TypeError):
                features["response_time"] = 0.0
        else:
            features["response_time"] = 0.0
        
        # Memory usage (if available)
        if "memory_usage" in log:
            try:
                features["memory_usage"] = float(log["memory_usage"])
            except (ValueError, TypeError):
                features["memory_usage"] = 0.0
        else:
            features["memory_usage"] = 0.0
        
        return features
    
    async def detect_anomalies_node(self, state: WorkflowState) -> WorkflowState:
        """Detect anomalies in the processed logs"""
        logger.info(f"Detecting anomalies in {len(state['processed_logs'])} logs")
        
        if len(state["processed_logs"]) < 10:
            logger.warning("Not enough logs for reliable anomaly detection")
            state["anomalies"] = []
            return state
        
        try:
            # Prepare feature matrix
            features_list = []
            for log in state["processed_logs"]:
                features = log["features"]
                feature_vector = [
                    features["message_length"],
                    features["error_keywords"],
                    features["level_severity"],
                    features["response_time"],
                    features["memory_usage"]
                ]
                features_list.append(feature_vector)
            
            # Convert to numpy array and scale
            X = np.array(features_list)
            X_scaled = self.scaler.fit_transform(X)
            
            # Detect anomalies using Isolation Forest
            anomaly_scores = self.isolation_forest.fit_predict(X_scaled)
            anomaly_probabilities = self.isolation_forest.score_samples(X_scaled)
            
            # Identify anomalies
            anomalies = []
            for i, (score, prob) in enumerate(zip(anomaly_scores, anomaly_probabilities)):
                if score == -1:  # Anomaly detected
                    confidence = abs(prob)  # Convert to positive confidence score
                    
                    if confidence >= state["confidence_threshold"]:
                        anomaly = {
                            "log_index": i,
                            "timestamp": state["processed_logs"][i]["timestamp"],
                            "level": state["processed_logs"][i]["level"],
                            "message": state["processed_logs"][i]["message"],
                            "service": state["processed_logs"][i]["service"],
                            "confidence": float(confidence),
                            "features": state["processed_logs"][i]["features"],
                            "anomaly_type": self._classify_anomaly_type(state["processed_logs"][i])
                        }
                        anomalies.append(anomaly)
            
            # Sort by confidence (highest first)
            anomalies.sort(key=lambda x: x["confidence"], reverse=True)
            
            state["anomalies"] = anomalies
            logger.info(f"Detected {len(anomalies)} anomalies")
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            state["anomalies"] = []
        
        return state
    
    def _classify_anomaly_type(self, log: Dict[str, Any]) -> str:
        """Classify the type of anomaly based on log content"""
        message = log.get("message", "").lower()
        features = log.get("features", {})
        
        # Performance anomaly
        if features.get("response_time", 0) > 5000:  # > 5 seconds
            return "performance"
        
        # Memory anomaly
        if features.get("memory_usage", 0) > 80:  # > 80% memory usage
            return "memory"
        
        # Error anomaly
        if features.get("error_keywords", 0) > 0 or log.get("level") in ["ERROR", "CRITICAL"]:
            return "error"
        
        # Pattern anomaly (unusual message patterns)
        if features.get("message_length", 0) > 1000:  # Very long messages
            return "pattern"
        
        return "unknown"
    
    async def format_results_node(self, state: WorkflowState) -> WorkflowState:
        """Format the final results"""
        logger.info(f"Formatting results for {len(state['anomalies'])} anomalies")
        
        # Add summary statistics
        for anomaly in state["anomalies"]:
            anomaly["analysis_timestamp"] = datetime.utcnow().isoformat() + 'Z'
        
        return state
    
    async def detect_anomalies(self, request: AnomalyRequest) -> AnomalyResponse:
        """Main method to detect anomalies for an incident"""
        logger.info(f"Starting anomaly detection for incident {request.incident_id}")
        
        # Initialize workflow state
        initial_state = WorkflowState(
            incident_id=request.incident_id,
            time_window_hours=request.time_window_hours,
            confidence_threshold=request.confidence_threshold,
            logs=[],
            processed_logs=[],
            anomalies=[],
            messages=[]
        )
        
        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        # Create response
        response = AnomalyResponse(
            incident_id=request.incident_id,
            anomalies_detected=len(final_state["anomalies"]),
            anomalies=final_state["anomalies"],
            processed_logs=len(final_state["processed_logs"]),
            analysis_timestamp=datetime.utcnow().isoformat() + 'Z'
        )
        
        logger.info(f"Anomaly detection completed for incident {request.incident_id}: {response.anomalies_detected} anomalies found")
        return response

# Global service instance
anomaly_service = AnomalyDetectionService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    await anomaly_service.initialize_redis()
    yield
    if anomaly_service.redis_client:
        await anomaly_service.redis_client.close()

# FastAPI app
app = FastAPI(
    title="Anomaly Detection Service",
    description="ML-powered anomaly detection for incident logs",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "anomaly-detection", "timestamp": datetime.utcnow().isoformat()}

@app.post("/detect-anomalies", response_model=AnomalyResponse)
async def detect_anomalies_endpoint(request: AnomalyRequest):
    """Detect anomalies in incident logs"""
    try:
        return await anomaly_service.detect_anomalies(request)
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
