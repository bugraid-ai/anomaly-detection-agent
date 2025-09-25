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
    confidence_scores: Dict[str, float]

class WorkflowState(TypedDict):
    """State for LangGraph workflow"""
    incident_id: str
    logs: List[Dict[str, Any]]
    processed_data: Optional[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    messages: Annotated[list[AnyMessage], add_messages]
    error: Optional[str]

class AnomalyDetectionService:
    """Core anomaly detection service"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
        # Anomaly thresholds
        self.thresholds = {
            'error_rate': 10.0,
            'warning_rate': 20.0,
            'pattern_count': 3.0,
            'error_severity': 3,
            'response_time_percentile': 95.0
        }
        
        self.redis_client = None
    
    async def connect_valkey(self):
        """Connect to Valkey"""
        try:
            self.redis_client = redis.Redis(
                host=VALKEY_HOST,
                port=VALKEY_PORT,
                username=VALKEY_USERNAME,
                password=VALKEY_PASSWORD,
                decode_responses=True,
                socket_timeout=10,
                socket_connect_timeout=10
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Successfully connected to Valkey")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Valkey: {e}")
            return False
    
    async def get_incident_logs(self, incident_id: str) -> List[Dict[str, Any]]:
        """Retrieve logs for a specific incident from Valkey"""
        try:
            if not self.redis_client:
                await self.connect_valkey()
            
            # Valkey key structure: bugraid-dev -> incidents -> bugraid-INC-XXX -> logs
            logs_key = f"bugraid-dev:incidents:{incident_id}:logs"
            
            # Get all log entries
            log_entries = await self.redis_client.lrange(logs_key, 0, -1)
            
            logs = []
            for entry in log_entries:
                try:
                    log_data = json.loads(entry)
                    logs.append(log_data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse log entry: {entry}")
                    continue
            
            logger.info(f"Retrieved {len(logs)} log entries for {incident_id}")
            return logs
            
        except Exception as e:
            logger.error(f"Error retrieving logs for {incident_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")
    
    def preprocess_logs(self, logs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess logs for anomaly detection"""
        if not logs:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(logs)
        
        # Extract key features for anomaly detection
        features = []
        for _, log in df.iterrows():
            feature_vector = self._extract_features(log)
            features.append(feature_vector)
        
        feature_df = pd.DataFrame(features)
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        return feature_df
    
    def _extract_features(self, log: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features from log entry"""
        features = {
            'log_level_numeric': self._log_level_to_numeric(log.get('level', 'INFO')),
            'message_length': len(str(log.get('message', ''))),
            'timestamp_hour': self._extract_hour(log.get('timestamp')),
            'error_code': self._extract_error_code(log.get('message', '')),
            'response_time': log.get('response_time', 0),
            'status_code': log.get('status_code', 200),
            'thread_count': log.get('thread_count', 1),
            'memory_usage': log.get('memory_usage', 0),
            'cpu_usage': log.get('cpu_usage', 0)
        }
        
        return features
    
    def _log_level_to_numeric(self, level: str) -> float:
        """Convert log level to numeric value"""
        level_map = {
            'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'WARNING': 3,
            'ERROR': 4, 'FATAL': 5, 'CRITICAL': 5
        }
        return level_map.get(level.upper(), 2)
    
    def _extract_hour(self, timestamp: Any) -> float:
        """Extract hour from timestamp"""
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return float(dt.hour)
            return 0.0
        except:
            return 0.0
    
    def _extract_error_code(self, message: str) -> float:
        """Extract error codes from message"""
        import re
        error_codes = re.findall(r'\b[45]\d{2}\b', str(message))
        return float(error_codes[0]) if error_codes else 0.0
    
    def detect_anomalies(self, feature_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalies using multiple methods"""
        if feature_df.empty:
            return []
        
        anomalies = []
        
        # Statistical anomaly detection
        stat_anomalies = self._statistical_anomaly_detection(feature_df)
        anomalies.extend(stat_anomalies)
        
        # ML-based anomaly detection
        if len(feature_df) >= 10:  # Need minimum samples for ML
            ml_anomalies = self._ml_anomaly_detection(feature_df)
            anomalies.extend(ml_anomalies)
        
        # Threshold-based anomaly detection
        threshold_anomalies = self._threshold_anomaly_detection(feature_df)
        anomalies.extend(threshold_anomalies)
        
        # Remove duplicates and rank by confidence
        unique_anomalies = self._deduplicate_anomalies(anomalies)
        
        return sorted(unique_anomalies, key=lambda x: x['confidence'], reverse=True)
    
    def _statistical_anomaly_detection(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Statistical-based anomaly detection using Z-score"""
        anomalies = []
        
        for column in df.select_dtypes(include=[np.number]).columns:
            mean_val = df[column].mean()
            std_val = df[column].std()
            
            if std_val == 0:
                continue
            
            z_scores = np.abs((df[column] - mean_val) / std_val)
            outlier_indices = np.where(z_scores > 3)[0]
            
            for idx in outlier_indices:
                anomalies.append({
                    'type': 'statistical',
                    'method': 'z_score',
                    'feature': column,
                    'value': df[column].iloc[idx],
                    'z_score': z_scores[idx],
                    'confidence': min(z_scores[idx] / 5.0, 1.0),
                    'index': idx,
                    'description': f'Statistical outlier in {column}'
                })
        
        return anomalies
    
    def _ml_anomaly_detection(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """ML-based anomaly detection using Isolation Forest"""
        anomalies = []
        
        try:
            # Prepare data
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return anomalies
            
            # Scale features
            scaled_data = self.scaler.fit_transform(numeric_df)
            
            # Fit and predict
            outliers = self.isolation_forest.fit_predict(scaled_data)
            scores = self.isolation_forest.score_samples(scaled_data)
            
            # Find anomalies
            for idx, (outlier, score) in enumerate(zip(outliers, scores)):
                if outlier == -1:  # Anomaly detected
                    confidence = 1.0 - (score + 0.5)  # Convert score to confidence
                    anomalies.append({
                        'type': 'ml',
                        'method': 'isolation_forest',
                        'score': score,
                        'confidence': max(0, min(confidence, 1.0)),
                        'index': idx,
                        'description': 'ML-detected anomaly using Isolation Forest'
                    })
        
        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
        
        return anomalies
    
    def _threshold_anomaly_detection(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Threshold-based anomaly detection"""
        anomalies = []
        
        # Error rate threshold
        if 'log_level_numeric' in df.columns:
            error_count = len(df[df['log_level_numeric'] >= 4])
            total_count = len(df)
            error_rate = (error_count / total_count * 100) if total_count > 0 else 0
            
            if error_rate > self.thresholds['error_rate']:
                anomalies.append({
                    'type': 'threshold',
                    'method': 'error_rate',
                    'value': error_rate,
                    'threshold': self.thresholds['error_rate'],
                    'confidence': min(error_rate / self.thresholds['error_rate'], 1.0),
                    'description': f'High error rate: {error_rate:.2f}%'
                })
        
        # Response time threshold
        if 'response_time' in df.columns:
            high_response_times = df['response_time'].quantile(0.95)
            if high_response_times > 5000:  # 5 seconds
                anomalies.append({
                    'type': 'threshold',
                    'method': 'response_time',
                    'value': high_response_times,
                    'threshold': 5000,
                    'confidence': min(high_response_times / 10000, 1.0),
                    'description': f'High response time (95th percentile): {high_response_times:.2f}ms'
                })
        
        return anomalies
    
    def _deduplicate_anomalies(self, anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate anomalies and merge similar ones"""
        unique_anomalies = []
        seen_signatures = set()
        
        for anomaly in anomalies:
            # Create signature for deduplication
            signature = f"{anomaly.get('type')}_{anomaly.get('method')}_{anomaly.get('feature', '')}"
            
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_anomalies.append(anomaly)
        
        return unique_anomalies

# Initialize service
anomaly_service = AnomalyDetectionService()

# LangGraph workflow definition
async def retrieve_logs_node(state: WorkflowState) -> WorkflowState:
    """Node to retrieve logs from Valkey"""
    try:
        incident_id = state["incident_id"]
        logs = await anomaly_service.get_incident_logs(incident_id)
        state["logs"] = logs
        state["messages"].append({"role": "system", "content": f"Retrieved {len(logs)} logs for {incident_id}"})
        return state
    except Exception as e:
        state["error"] = str(e)
        return state

async def process_logs_node(state: WorkflowState) -> WorkflowState:
    """Node to process logs and extract features"""
    try:
        logs = state["logs"]
        feature_df = anomaly_service.preprocess_logs(logs)
        state["processed_data"] = {
            "feature_count": len(feature_df.columns),
            "log_count": len(feature_df),
            "features": feature_df.to_dict('records') if len(feature_df) < 100 else []
        }
        state["messages"].append({"role": "system", "content": f"Processed {len(logs)} logs into {len(feature_df.columns)} features"})
        return state
    except Exception as e:
        state["error"] = str(e)
        return state

async def detect_anomalies_node(state: WorkflowState) -> WorkflowState:
    """Node to detect anomalies"""
    try:
        logs = state["logs"]
        feature_df = anomaly_service.preprocess_logs(logs)
        anomalies = anomaly_service.detect_anomalies(feature_df)
        state["anomalies"] = anomalies
        state["messages"].append({"role": "system", "content": f"Detected {len(anomalies)} anomalies"})
        return state
    except Exception as e:
        state["error"] = str(e)
        return state

# Create LangGraph workflow
def create_anomaly_workflow():
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("retrieve_logs", retrieve_logs_node)
    workflow.add_node("process_logs", process_logs_node) 
    workflow.add_node("detect_anomalies", detect_anomalies_node)
    
    # Add edges
    workflow.add_edge(START, "retrieve_logs")
    workflow.add_edge("retrieve_logs", "process_logs")
    workflow.add_edge("process_logs", "detect_anomalies")
    workflow.add_edge("detect_anomalies", END)
    
    return workflow.compile()

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Anomaly Detection API")
    await anomaly_service.connect_valkey()
    yield
    # Shutdown
    if anomaly_service.redis_client:
        await anomaly_service.redis_client.close()
    logger.info("Anomaly Detection API shutdown complete")

# FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="FastAPI service for detecting anomalies in incident logs using LangGraph workflows",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "anomaly-detection",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "valkey_connected": anomaly_service.redis_client is not None
    }

@app.post("/detect-anomalies", response_model=AnomalyResponse)
async def detect_anomalies(request: AnomalyRequest, background_tasks: BackgroundTasks):
    """Detect anomalies in incident logs"""
    try:
        # Create workflow
        workflow = create_anomaly_workflow()
        
        # Initial state
        initial_state = WorkflowState(
            incident_id=request.incident_id,
            logs=[],
            processed_data=None,
            anomalies=[],
            messages=[],
            error=None
        )
        
        # Run workflow
        result = await workflow.ainvoke(initial_state)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Calculate confidence scores
        anomalies = result["anomalies"]
        confidence_scores = {}
        
        if anomalies:
            confidence_scores = {
                "average_confidence": sum(a["confidence"] for a in anomalies) / len(anomalies),
                "max_confidence": max(a["confidence"] for a in anomalies),
                "min_confidence": min(a["confidence"] for a in anomalies)
            }
        
        # Filter by confidence threshold
        filtered_anomalies = [
            a for a in anomalies 
            if a["confidence"] >= request.confidence_threshold
        ]
        
        return AnomalyResponse(
            incident_id=request.incident_id,
            anomalies_detected=len(filtered_anomalies),
            anomalies=filtered_anomalies,
            processed_logs=len(result["logs"]),
            analysis_timestamp=datetime.utcnow().isoformat() + "Z",
            confidence_scores=confidence_scores
        )
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/incident/{incident_id}/logs")
async def get_incident_logs(incident_id: str):
    """Get raw logs for an incident"""
    try:
        logs = await anomaly_service.get_incident_logs(incident_id)
        return {
            "incident_id": incident_id,
            "log_count": len(logs),
            "logs": logs[:100],  # Limit response size
            "truncated": len(logs) > 100
        }
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
