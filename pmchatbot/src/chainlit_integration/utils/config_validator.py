"""
Configuration validation and setup instructions for Chainlit integration.

This module provides comprehensive configuration validation, setup verification,
and user-friendly setup instructions for the process mining analysis tool.

Features:
- Environment validation
- Dependency checking
- Configuration file validation
- Setup instructions generation
- Health checks
- Troubleshooting guidance
"""

import os
import sys
import logging
import importlib
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    component: str
    status: str  # 'pass', 'fail', 'warning'
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None


@dataclass
class SystemRequirements:
    """System requirements specification."""
    python_min_version: Tuple[int, int] = (3, 7)
    python_max_version: Tuple[int, int] = (3, 12)
    required_packages: List[str] = None
    optional_packages: List[str] = None
    required_files: List[str] = None
    environment_variables: List[str] = None
    
    def __post_init__(self):
        if self.required_packages is None:
            self.required_packages = [
                "chainlit>=2.6.0",
                "pandas>=2.0.0",
                "pm4py>=2.7.0",
                "neo4j>=5.0.0",
                "python-dotenv>=1.0.0",
                "asyncio",
                "logging"
            ]
        
        if self.optional_packages is None:
            self.optional_packages = [
                "openai>=1.0.0",
                "torch>=2.0.0",
                "sentence-transformers>=2.0.0",
                "matplotlib>=3.0.0",
                "networkx>=3.0.0"
            ]
        
        if self.required_files is None:
            self.required_files = [
                "Production_Event_Log.csv",
                ".env.template"
            ]
        
        if self.environment_variables is None:
            self.environment_variables = [
                "NEO4J_URI",
                "NEO4J_USERNAME", 
                "NEO4J_PASSWORD"
            ]


class ConfigurationValidator:
    """Validate system configuration and setup."""
    
    def __init__(self, requirements: SystemRequirements = None):
        self.requirements = requirements or SystemRequirements()
        self.validation_results: List[ValidationResult] = []
    
    def validate_all(self) -> List[ValidationResult]:
        """Run all validation checks."""
        self.validation_results.clear()
        
        # Run all validation checks
        self._validate_python_version()
        self._validate_required_packages()
        self._validate_optional_packages()
        self._validate_required_files()
        self._validate_environment_variables()
        self._validate_neo4j_connection()
        self._validate_csv_data()
        self._validate_directory_structure()
        self._validate_permissions()
        
        return self.validation_results
    
    def _validate_python_version(self):
        """Validate Python version."""
        current_version = sys.version_info[:2]
        min_version = self.requirements.python_min_version
        max_version = self.requirements.python_max_version
        
        if current_version < min_version:
            self.validation_results.append(ValidationResult(
                component="Python Version",
                status="fail",
                message=f"Python {current_version[0]}.{current_version[1]} is too old",
                details=f"Minimum required: {min_version[0]}.{min_version[1]}",
                fix_suggestion=f"Upgrade Python to version {min_version[0]}.{min_version[1]} or higher"
            ))
        elif current_version > max_version:
            self.validation_results.append(ValidationResult(
                component="Python Version",
                status="warning",
                message=f"Python {current_version[0]}.{current_version[1]} is newer than tested",
                details=f"Tested up to: {max_version[0]}.{max_version[1]}",
                fix_suggestion="Consider using a tested Python version if you encounter issues"
            ))
        else:
            self.validation_results.append(ValidationResult(
                component="Python Version",
                status="pass",
                message=f"Python {current_version[0]}.{current_version[1]} is compatible"
            ))
    
    def _validate_required_packages(self):
        """Validate required packages."""
        for package_spec in self.requirements.required_packages:
            package_name = package_spec.split(">=")[0].split("==")[0]
            
            try:
                importlib.import_module(package_name)
                self.validation_results.append(ValidationResult(
                    component=f"Package: {package_name}",
                    status="pass",
                    message=f"{package_name} is installed"
                ))
            except ImportError:
                self.validation_results.append(ValidationResult(
                    component=f"Package: {package_name}",
                    status="fail",
                    message=f"{package_name} is not installed",
                    details=f"Required: {package_spec}",
                    fix_suggestion=f"Install with: pip install {package_spec}"
                ))
    
    def _validate_optional_packages(self):
        """Validate optional packages."""
        for package_spec in self.requirements.optional_packages:
            package_name = package_spec.split(">=")[0].split("==")[0]
            
            try:
                importlib.import_module(package_name)
                self.validation_results.append(ValidationResult(
                    component=f"Optional Package: {package_name}",
                    status="pass",
                    message=f"{package_name} is available"
                ))
            except ImportError:
                self.validation_results.append(ValidationResult(
                    component=f"Optional Package: {package_name}",
                    status="warning",
                    message=f"{package_name} is not installed",
                    details="This package provides enhanced functionality",
                    fix_suggestion=f"Install with: pip install {package_spec}"
                ))
    
    def _validate_required_files(self):
        """Validate required files exist."""
        for file_path in self.requirements.required_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.validation_results.append(ValidationResult(
                    component=f"File: {file_path}",
                    status="pass",
                    message=f"{file_path} exists ({file_size:,} bytes)"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component=f"File: {file_path}",
                    status="fail",
                    message=f"{file_path} not found",
                    fix_suggestion=f"Ensure {file_path} is in the correct location"
                ))
    
    def _validate_environment_variables(self):
        """Validate environment variables."""
        # Check for .env file
        env_file = Path(".env")
        if env_file.exists():
            self.validation_results.append(ValidationResult(
                component="Environment File",
                status="pass",
                message=".env file found"
            ))
        else:
            self.validation_results.append(ValidationResult(
                component="Environment File",
                status="warning",
                message=".env file not found",
                fix_suggestion="Copy .env.template to .env and configure values"
            ))
        
        # Check individual environment variables
        for env_var in self.requirements.environment_variables:
            value = os.getenv(env_var)
            if value:
                # Mask sensitive values
                display_value = value if env_var not in ["NEO4J_PASSWORD", "OPENAI_API_KEY"] else "***"
                self.validation_results.append(ValidationResult(
                    component=f"Environment: {env_var}",
                    status="pass",
                    message=f"{env_var} is set",
                    details=f"Value: {display_value}"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component=f"Environment: {env_var}",
                    status="warning",
                    message=f"{env_var} is not set",
                    fix_suggestion=f"Set {env_var} in your .env file"
                ))
    
    def _validate_neo4j_connection(self):
        """Validate Neo4j connection."""
        try:
            from neo4j import GraphDatabase
            
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            username = os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD")
            
            if not password:
                self.validation_results.append(ValidationResult(
                    component="Neo4j Connection",
                    status="warning",
                    message="Neo4j password not configured",
                    fix_suggestion="Set NEO4J_PASSWORD in .env file"
                ))
                return
            
            # Test connection
            driver = GraphDatabase.driver(uri, auth=(username, password))
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
                if test_value == 1:
                    self.validation_results.append(ValidationResult(
                        component="Neo4j Connection",
                        status="pass",
                        message="Neo4j connection successful",
                        details=f"Connected to {uri}"
                    ))
                else:
                    self.validation_results.append(ValidationResult(
                        component="Neo4j Connection",
                        status="fail",
                        message="Neo4j connection test failed"
                    ))
            
            driver.close()
            
        except ImportError:
            self.validation_results.append(ValidationResult(
                component="Neo4j Connection",
                status="fail",
                message="Neo4j driver not installed",
                fix_suggestion="Install with: pip install neo4j"
            ))
        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="Neo4j Connection",
                status="fail",
                message="Neo4j connection failed",
                details=str(e),
                fix_suggestion="Check Neo4j server is running and credentials are correct"
            ))
    
    def _validate_csv_data(self):
        """Validate CSV data file."""
        csv_file = "Production_Event_Log.csv"
        
        if not os.path.exists(csv_file):
            self.validation_results.append(ValidationResult(
                component="CSV Data",
                status="fail",
                message="CSV data file not found",
                fix_suggestion=f"Place your process data CSV file as {csv_file}"
            ))
            return
        
        try:
            import pandas as pd
            
            # Read a sample of the CSV
            df = pd.read_csv(csv_file, nrows=100)
            
            # Check required columns
            required_columns = ["case_id", "activity", "timestamp", "part_desc"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.validation_results.append(ValidationResult(
                    component="CSV Data Structure",
                    status="fail",
                    message="CSV missing required columns",
                    details=f"Missing: {', '.join(missing_columns)}",
                    fix_suggestion=f"Ensure CSV has columns: {', '.join(required_columns)}"
                ))
            else:
                # Get data statistics
                total_rows = len(pd.read_csv(csv_file))
                unique_parts = df["part_desc"].nunique()
                unique_activities = df["activity"].nunique()
                
                self.validation_results.append(ValidationResult(
                    component="CSV Data Structure",
                    status="pass",
                    message="CSV data structure is valid",
                    details=f"Rows: {total_rows:,}, Parts: {unique_parts}, Activities: {unique_activities}"
                ))
        
        except ImportError:
            self.validation_results.append(ValidationResult(
                component="CSV Data",
                status="fail",
                message="Pandas not available for CSV validation",
                fix_suggestion="Install pandas: pip install pandas"
            ))
        except Exception as e:
            self.validation_results.append(ValidationResult(
                component="CSV Data",
                status="fail",
                message="CSV data validation failed",
                details=str(e),
                fix_suggestion="Check CSV file format and content"
            ))
    
    def _validate_directory_structure(self):
        """Validate directory structure."""
        required_dirs = [
            "src",
            "src/chainlit_integration",
            "src/chainlit_integration/managers",
            "src/chainlit_integration/utils",
            "src/data"
        ]
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                self.validation_results.append(ValidationResult(
                    component=f"Directory: {dir_path}",
                    status="pass",
                    message=f"{dir_path} exists"
                ))
            else:
                self.validation_results.append(ValidationResult(
                    component=f"Directory: {dir_path}",
                    status="fail",
                    message=f"{dir_path} not found",
                    fix_suggestion=f"Create directory: {dir_path}"
                ))
    
    def _validate_permissions(self):
        """Validate file permissions."""
        # Check write permissions for temp directories
        temp_dirs = ["uploads", ".files", "temp"]
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                if os.access(temp_dir, os.W_OK):
                    self.validation_results.append(ValidationResult(
                        component=f"Permissions: {temp_dir}",
                        status="pass",
                        message=f"{temp_dir} is writable"
                    ))
                else:
                    self.validation_results.append(ValidationResult(
                        component=f"Permissions: {temp_dir}",
                        status="fail",
                        message=f"{temp_dir} is not writable",
                        fix_suggestion=f"Fix permissions for {temp_dir}"
                    ))
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        total = len(self.validation_results)
        passed = len([r for r in self.validation_results if r.status == "pass"])
        failed = len([r for r in self.validation_results if r.status == "fail"])
        warnings = len([r for r in self.validation_results if r.status == "warning"])
        
        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "ready_to_run": failed == 0
        }


class SetupInstructionsGenerator:
    """Generate setup instructions based on validation results."""
    
    def __init__(self, validator: ConfigurationValidator):
        self.validator = validator
    
    def generate_setup_instructions(self) -> str:
        """Generate comprehensive setup instructions."""
        instructions = """# 🚀 Process Mining Analysis Tool - Setup Instructions

## Prerequisites

### 1. Python Environment
- **Python Version**: 3.7 - 3.11 (3.9 recommended)
- **Package Manager**: pip (included with Python)

### 2. Neo4j Database
- **Neo4j Desktop** (recommended): https://neo4j.com/download/
- **Or Neo4j Community Server**: https://neo4j.com/deployment-center/

## Installation Steps

### Step 1: Clone and Setup Project
```bash
# Clone the repository (if applicable)
git clone <repository-url>
cd pmchatbot

# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Or install individually:
pip install chainlit>=2.6.0
pip install pandas>=2.0.0
pip install pm4py>=2.7.0
pip install neo4j>=5.0.0
pip install python-dotenv>=1.0.0
pip install openai>=1.0.0  # Optional, for OpenAI integration
```

### Step 3: Setup Neo4j Database
1. **Install Neo4j Desktop** from https://neo4j.com/download/
2. **Create a new database** with these settings:
   - Name: `process-mining`
   - Password: Choose a secure password
   - Version: 5.x (latest)
3. **Start the database**
4. **Note the connection details** (usually `bolt://localhost:7687`)

### Step 4: Configure Environment
1. **Copy environment template**:
   ```bash
   cp .env.template .env
   ```

2. **Edit .env file** with your settings:
   ```env
   # Neo4j Configuration
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password_here
   
   # Optional: OpenAI Configuration
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional: Logging
   LOG_LEVEL=INFO
   CHAINLIT_DEBUG=false
   ```

### Step 5: Prepare Data
1. **Place your CSV file** as `Production_Event_Log.csv` in the project root
2. **Ensure CSV has required columns**:
   - `case_id`: Unique identifier for each process case
   - `activity`: Name of the activity/step
   - `timestamp`: When the activity occurred
   - `part_desc`: Description of the part/product being processed

### Step 6: Verify Installation
```bash
# Run configuration validation
python -c "from src.chainlit_integration.utils.config_validator import ConfigurationValidator; validator = ConfigurationValidator(); results = validator.validate_all(); print('Setup validation complete')"

# Or run the validation script
python src/chainlit_integration/utils/config_validator.py
```

### Step 7: Start the Application
```bash
# Start Chainlit application
chainlit run src/chainlit_app.py

# Or with custom port
chainlit run src/chainlit_app.py --port 8001
```

## Troubleshooting

### Common Issues

**1. Neo4j Connection Failed**
- Ensure Neo4j database is running
- Check connection details in .env file
- Verify firewall settings allow connection to port 7687

**2. CSV Data Issues**
- Verify CSV file exists and has correct name
- Check that required columns are present
- Ensure timestamp format is readable by pandas

**3. Package Installation Issues**
- Update pip: `pip install --upgrade pip`
- Use virtual environment to avoid conflicts
- On Windows, you may need Visual Studio Build Tools for some packages

**4. Permission Errors**
- Ensure write permissions for temp directories
- Run with appropriate user permissions
- Check antivirus software isn't blocking file operations

### Getting Help

1. **Check logs**: Look at `chainlit_app.log` for detailed error messages
2. **Run validation**: Use the configuration validator to identify issues
3. **Debug mode**: Set `CHAINLIT_DEBUG=true` in .env for detailed logging
4. **Community support**: Check Chainlit and PM4py documentation

## Optional Enhancements

### Ollama Setup (for local LLM)
1. **Install Ollama**: https://ollama.ai/
2. **Pull a model**: `ollama pull llama2`
3. **Verify**: `ollama list`

### Performance Optimization
- **Increase Neo4j memory**: Edit Neo4j configuration
- **Use SSD storage**: For better I/O performance
- **Monitor resources**: Use task manager to monitor CPU/memory usage

## Next Steps

Once setup is complete:
1. **Start the application**: `chainlit run src/chainlit_app.py`
2. **Open browser**: Navigate to http://localhost:8000
3. **Select LLM**: Choose between OpenAI or Ollama
4. **Select data**: Choose a part/product to analyze
5. **Ask questions**: Start exploring your process data!

---

**Need help?** Check the troubleshooting section or run the configuration validator for specific guidance.
"""
        
        return instructions
    
    def generate_quick_start_guide(self) -> str:
        """Generate a quick start guide."""
        return """# ⚡ Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies (2 minutes)
```bash
pip install chainlit pandas pm4py neo4j python-dotenv
```

### 2. Setup Neo4j (2 minutes)
- Download Neo4j Desktop
- Create database with password
- Start the database

### 3. Configure Environment (1 minute)
```bash
cp .env.template .env
# Edit .env with your Neo4j password
```

### 4. Add Your Data
- Place CSV file as `Production_Event_Log.csv`
- Ensure columns: case_id, activity, timestamp, part_desc

### 5. Run Application
```bash
chainlit run src/chainlit_app.py
```

**That's it!** Open http://localhost:8000 and start analyzing your process data.

---

**Having issues?** Run the full setup validation:
```bash
python src/chainlit_integration/utils/config_validator.py
```
"""
    
    def generate_troubleshooting_guide(self, validation_results: List[ValidationResult]) -> str:
        """Generate troubleshooting guide based on validation results."""
        failed_checks = [r for r in validation_results if r.status == "fail"]
        warning_checks = [r for r in validation_results if r.status == "warning"]
        
        if not failed_checks and not warning_checks:
            return "✅ **All checks passed!** Your system is ready to run the application."
        
        guide = "# 🔧 Troubleshooting Guide\n\n"
        
        if failed_checks:
            guide += "## ❌ Critical Issues (Must Fix)\n\n"
            for result in failed_checks:
                guide += f"### {result.component}\n"
                guide += f"**Problem**: {result.message}\n"
                if result.details:
                    guide += f"**Details**: {result.details}\n"
                if result.fix_suggestion:
                    guide += f"**Solution**: {result.fix_suggestion}\n"
                guide += "\n"
        
        if warning_checks:
            guide += "## ⚠️ Warnings (Recommended Fixes)\n\n"
            for result in warning_checks:
                guide += f"### {result.component}\n"
                guide += f"**Issue**: {result.message}\n"
                if result.details:
                    guide += f"**Details**: {result.details}\n"
                if result.fix_suggestion:
                    guide += f"**Recommendation**: {result.fix_suggestion}\n"
                guide += "\n"
        
        return guide


def main():
    """Main function for running validation."""
    print("🔍 Process Mining Analysis Tool - Configuration Validation")
    print("=" * 60)
    
    # Run validation
    validator = ConfigurationValidator()
    results = validator.validate_all()
    
    # Print results
    for result in results:
        status_emoji = {"pass": "✅", "fail": "❌", "warning": "⚠️"}[result.status]
        print(f"{status_emoji} {result.component}: {result.message}")
        
        if result.details:
            print(f"   Details: {result.details}")
        if result.fix_suggestion:
            print(f"   Fix: {result.fix_suggestion}")
        print()
    
    # Print summary
    summary = validator.get_validation_summary()
    print("=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Checks: {summary['total_checks']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⚠️ Warnings: {summary['warnings']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Ready to Run: {'Yes' if summary['ready_to_run'] else 'No'}")
    
    # Generate setup instructions if needed
    if not summary['ready_to_run']:
        print("\n" + "=" * 60)
        print("🚀 SETUP REQUIRED")
        print("=" * 60)
        
        generator = SetupInstructionsGenerator(validator)
        troubleshooting = generator.generate_troubleshooting_guide(results)
        print(troubleshooting)
        
        print("\nFor complete setup instructions, run:")
        print("python -c \"from src.chainlit_integration.utils.config_validator import SetupInstructionsGenerator, ConfigurationValidator; generator = SetupInstructionsGenerator(ConfigurationValidator()); print(generator.generate_setup_instructions())\"")


if __name__ == "__main__":
    main()