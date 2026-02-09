#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Student Success Pipeline - Master Controller
============================================
Execute complete student success prediction pipeline

Pipeline Stages:
    Stage 1: Data Quality Assessment and Cleaning
    Stage 2: Feature Engineering
    Stage 3: Modeling and Action Generation
    Stage 4: Integrated Report Generation

Usage:
    python run_master_pipeline.py
    python run_master_pipeline.py --skip-stage1
    python run_master_pipeline.py --stage 2
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
import traceback

# 添加项目根目录到系统路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class PipelineOrchestrator:
    """Pipeline Orchestrator - Coordinates execution of all stages"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline orchestrator
        
        Args:
            config_path: Path to configuration file
        """
        self.project_root = PROJECT_ROOT
        self.config_path = self.project_root / config_path
        self.config = self._load_config()
        self.logger = self._setup_logger()
        
        # Define output paths for each stage
        self.paths = {
            'stage1_output': self.project_root / 'outputs' / 'stage1_quality' / 'pipeline_data.json',
            'stage2_output': self.project_root / 'outputs' / 'stage2_features' / 'feature_strategy.json',
            'stage3_output': self.project_root / 'outputs' / 'stage3_modeling' / 'modeling_results.json',
            'stage4_output': self.project_root / 'outputs' / 'reports' / 'final_report.html',
        }
        
        # Track execution status
        self.execution_status = {
            'stage1': {'status': 'pending', 'duration': None, 'error': None},
            'stage2': {'status': 'pending', 'duration': None, 'error': None},
            'stage3': {'status': 'pending', 'duration': None, 'error': None},
            'stage4': {'status': 'pending', 'duration': None, 'error': None},
        }
        
    def _load_config(self) -> dict:
        """Load configuration file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found: {self.config_path}")
            print("Continuing with default configuration...")
            return {}
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging system"""
        # Create logs directory
        log_dir = self.project_root / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # Log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'pipeline_run_{timestamp}.log'
        
        # Configure logger
        logger = logging.getLogger('PipelineMaster')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _print_banner(self):
        """Print welcome banner"""
        banner = """
================================================================
          Education Analytics Pipeline
================================================================
"""
        print(banner)
        self.logger.info("=" * 70)
        self.logger.info("Pipeline started: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.logger.info("=" * 70)
    
    def _print_stage_header(self, stage_num: int, stage_name: str):
        """Print stage header"""
        header = f"""
================================================================
  Stage {stage_num}: {stage_name}
================================================================
"""
        print(header)
        self.logger.info(f"Starting Stage {stage_num}: {stage_name}")
    
    def run_stage1_data_quality(self) -> bool:
        """
        Stage 1: Data Quality Assessment and Cleaning
        
        Returns:
            bool: Whether execution was successful
        """
        self._print_stage_header(1, "Data Quality Assessment and Cleaning")
        start_time = datetime.now()
        
        try:
            # Import stage 1 main function
            from src.stage1_data_quality.run_data_quality import main as stage1_main
            
            # Execute stage 1
            self.logger.info("Running data quality assessment pipeline...")
            stage1_main()
            
            # Check output file
            if not self.paths['stage1_output'].exists():
                raise FileNotFoundError(f"Stage 1 output not generated: {self.paths['stage1_output']}")
            
            # Update status
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_status['stage1'] = {
                'status': 'success',
                'duration': duration,
                'error': None
            }
            
            self.logger.info(f"Stage 1 completed (Duration: {duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            self.execution_status['stage1'] = {
                'status': 'failed',
                'duration': duration,
                'error': error_msg
            }
            
            self.logger.error(f"Stage 1 failed: {error_msg}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_stage2_feature_engineering(self) -> bool:
        """
        Stage 2: Feature Engineering
        
        Returns:
            bool: Whether execution was successful
        """
        self._print_stage_header(2, "Feature Engineering")
        start_time = datetime.now()
        
        try:
            # Check dependencies
            if not self.paths['stage1_output'].exists():
                raise FileNotFoundError("Stage 1 output not found. Please run Stage 1 first.")
            
            # Import stage 2 main function
            from src.stage2_feature_engineering.run_feature_strategy import main as stage2_main
            
            # Execute stage 2
            self.logger.info("Running feature engineering pipeline...")
            stage2_main()
            
            # Check output file
            if not self.paths['stage2_output'].exists():
                raise FileNotFoundError(f"Stage 2 output not generated: {self.paths['stage2_output']}")
            
            # Update status
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_status['stage2'] = {
                'status': 'success',
                'duration': duration,
                'error': None
            }
            
            self.logger.info(f"Stage 2 completed (Duration: {duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            self.execution_status['stage2'] = {
                'status': 'failed',
                'duration': duration,
                'error': error_msg
            }
            
            self.logger.error(f"Stage 2 failed: {error_msg}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_stage3_modeling_action(self) -> bool:
        """
        Stage 3: Modeling and Action Plan Generation
        
        Returns:
            bool: Whether execution was successful
        """
        self._print_stage_header(3, "Modeling and Action Plan Generation")
        start_time = datetime.now()
        
        try:
            # Check dependencies
            if not self.paths['stage2_output'].exists():
                raise FileNotFoundError("Stage 2 output not found. Please run Stage 2 first.")
            
            # Import stage 3 main function
            from src.stage3_modeling_action.run_modeling_action import main as stage3_main
            
            # Execute stage 3
            self.logger.info("Running modeling and action plan generation pipeline...")
            stage3_main()
            
            # Check output file
            if not self.paths['stage3_output'].exists():
                raise FileNotFoundError(f"Stage 3 output not generated: {self.paths['stage3_output']}")
            
            # Update status
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_status['stage3'] = {
                'status': 'success',
                'duration': duration,
                'error': None
            }
            
            self.logger.info(f"Stage 3 completed (Duration: {duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            self.execution_status['stage3'] = {
                'status': 'failed',
                'duration': duration,
                'error': error_msg
            }
            
            self.logger.error(f"Stage 3 failed: {error_msg}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_stage4_reporting(self) -> bool:
        """
        Stage 4: Integrated Report Generation
        
        Returns:
            bool: Whether execution was successful
        """
        self._print_stage_header(4, "Integrated Report Generation")
        start_time = datetime.now()
        
        try:
            # Check dependencies
            dependencies = [
                self.paths['stage1_output'],
                self.paths['stage2_output'],
                self.paths['stage3_output']
            ]
            
            for dep_path in dependencies:
                if not dep_path.exists():
                    raise FileNotFoundError(f"Dependency file not found: {dep_path}")
            
            # Import stage 4 main function
            from src.stage4_reporting.build_integrated_report import main as stage4_main
            
            # Execute stage 4
            self.logger.info("Generating integrated report...")
            stage4_main()
            
            # Check output file
            if not self.paths['stage4_output'].exists():
                raise FileNotFoundError(f"Stage 4 output not generated: {self.paths['stage4_output']}")
            
            # Update status
            duration = (datetime.now() - start_time).total_seconds()
            self.execution_status['stage4'] = {
                'status': 'success',
                'duration': duration,
                'error': None
            }
            
            self.logger.info(f"Stage 4 completed (Duration: {duration:.2f}s)")
            self.logger.info(f"Report generated: {self.paths['stage4_output']}")
            return True
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            self.execution_status['stage4'] = {
                'status': 'failed',
                'duration': duration,
                'error': error_msg
            }
            
            self.logger.error(f"Stage 4 failed: {error_msg}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def run_full_pipeline(self, start_stage: int = 1, end_stage: int = 4) -> bool:
        """
        Run complete pipeline or specified stages
        
        Args:
            start_stage: Starting stage (1-4)
            end_stage: Ending stage (1-4)
            
        Returns:
            bool: Whether all stages executed successfully
        """
        self._print_banner()
        
        # Define stage execution function mapping
        stage_functions = {
            1: self.run_stage1_data_quality,
            2: self.run_stage2_feature_engineering,
            3: self.run_stage3_modeling_action,
            4: self.run_stage4_reporting,
        }
        
        overall_start = datetime.now()
        all_success = True
        
        # Execute specified stage range
        for stage_num in range(start_stage, end_stage + 1):
            if stage_num in stage_functions:
                success = stage_functions[stage_num]()
                if not success:
                    all_success = False
                    self.logger.warning(f"Stage {stage_num} failed. Continue with subsequent stages?")
                print()
        
        # Print summary
        self._print_summary(overall_start)
        
        return all_success
    
    def _print_summary(self, start_time: datetime):
        """Print execution summary"""
        total_duration = (datetime.now() - start_time).total_seconds()
        
        summary = """
================================================================
                    Execution Summary
================================================================
"""
        print(summary)
        
        # Stage status
        for stage_name, status_info in self.execution_status.items():
            status = status_info['status']
            duration = status_info['duration']
            
            if status == 'success':
                status_text = "SUCCESS"
            elif status == 'failed':
                status_text = "FAILED"
            else:
                status_text = "SKIPPED"
            
            duration_text = f"{duration:.2f}s" if duration else "N/A"
            print(f"{stage_name.upper():<10} [{status_text:<8}]  Duration: {duration_text}")
            
            if status_info['error']:
                print(f"  Error: {status_info['error']}")
        
        print()
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Output file locations
        print("Output Files:")
        for key, path in self.paths.items():
            if path.exists():
                print(f"  {key}: {path.relative_to(self.project_root)}")
            else:
                print(f"  {key}: Not generated")
        
        print()
        self.logger.info("=" * 70)
        self.logger.info(f"Pipeline completed. Total duration: {total_duration:.2f}s")
        self.logger.info("=" * 70)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Student Success Prediction Pipeline - Master Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python run_master_pipeline.py                    # Run complete pipeline
  python run_master_pipeline.py --stage 2          # Run only stage 2
  python run_master_pipeline.py --start 2 --end 4  # Run stages 2-4
  python run_master_pipeline.py --skip-stage1      # Skip stage 1
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )
    
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run only specified stage'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help='Starting stage (default: 1)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help='Ending stage (default: 4)'
    )
    
    parser.add_argument(
        '--skip-stage1',
        action='store_true',
        help='Skip stage 1 (data quality)'
    )
    
    parser.add_argument(
        '--skip-stage2',
        action='store_true',
        help='Skip stage 2 (feature engineering)'
    )
    
    parser.add_argument(
        '--skip-stage3',
        action='store_true',
        help='Skip stage 3 (modeling)'
    )
    
    parser.add_argument(
        '--skip-stage4',
        action='store_true',
        help='Skip stage 4 (reporting)'
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create pipeline orchestrator
    orchestrator = PipelineOrchestrator(config_path=args.config)
    
    # Determine execution range
    if args.stage:
        # Run single stage only
        start_stage = args.stage
        end_stage = args.stage
    else:
        # Run stage range
        start_stage = args.start
        end_stage = args.end
        
        # Handle skip options
        if args.skip_stage1 and start_stage == 1:
            start_stage = 2
        if args.skip_stage4 and end_stage == 4:
            end_stage = 3
    
    # Validate stage range
    if start_stage > end_stage:
        print(f"Error: Starting stage ({start_stage}) cannot be greater than ending stage ({end_stage})")
        sys.exit(1)
    
    try:
        # Run pipeline
        success = orchestrator.run_full_pipeline(start_stage, end_stage)
        
        # Return exit code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        orchestrator.logger.warning("Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        orchestrator.logger.error(f"Unexpected error: {e}")
        orchestrator.logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
