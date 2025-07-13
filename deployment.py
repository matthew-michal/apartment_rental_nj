from prefect import flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect_aws import AwsBatch
from main_flow import daily_prediction_flow
from model_training import weekly_model_training_flow

def create_deployments():
    """
    Create and apply deployments for both daily prediction and weekly training flows.
    """
    # Create AWS Batch block 
    aws_batch_block = AwsBatch(
        job_queue="your-job-queue",
        job_definition="your-job-definition",
        region="us-east-1"
    )
    
    # Create daily prediction deployment
    daily_deployment = Deployment.build_from_flow(
        flow=daily_prediction_flow,
        name="daily-prediction-deployment",
        schedule=CronSchedule(cron="0 7 * * *", timezone="America/New_York"),  # 7 AM daily
        infrastructure=aws_batch_block,
        parameters={},
        work_pool_name="default-work-pool"
    )
    
    # Create weekly training deployment
    weekly_deployment = Deployment.build_from_flow(
        flow=weekly_model_training_flow,
        name="weekly-training-deployment",
        schedule=CronSchedule(cron="0 0 * * 0", timezone="America/New_York"),  # Midnight on Sundays
        infrastructure=aws_batch_block,
        parameters={},
        work_pool_name="default-work-pool"
    )
    
    # Apply deployments
    daily_deployment.apply()
    weekly_deployment.apply()
    
    print("Both deployments created successfully!")
    print("Daily prediction: 7 AM daily")
    print("Weekly training: Midnight on Sundays")

if __name__ == "__main__":
    create_deployments()