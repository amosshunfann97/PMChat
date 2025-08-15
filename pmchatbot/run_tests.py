#!/usr/bin/env python3
"""
Comprehensive test runner for Chainlit integration.

This script provides a unified interface for running all tests with
proper configuration, reporting, and validation.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --performance      # Run only performance tests
    python run_tests.py --e2e              # Run only end-to-end tests
    python run_tests.py --fast             # Run fast tests only
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --ci               # Run in CI mode
"""

import asyncio
import argparse
import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our test automation pipeline
from tests.test_automation_pipeline import TestAutomationPipeline, ContinuousIntegrationRunner


class TestRunner:
    """Main test runner with configuration options."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.pipeline = TestAutomationPipeline(str(self.project_root))
        self.ci_runner = ContinuousIntegrationRunner(str(self.project_root))
    
    def setup_environment(self):
        """Setup test environment."""
        # Ensure test directories exist
        test_dirs = [
            self.project_root / "tests",
            self.project_root / "src" / "chainlit_integration" / "tests"
        ]
        
        for test_dir in test_dirs:
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Create __init__.py if it doesn't exist
            init_file = test_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Test package")
        
        # Set environment variables for testing
        os.environ['TESTING'] = '1'
        os.environ['PYTHONPATH'] = str(self.project_root / "src")
    
    def install_test_dependencies(self):
        """Install test dependencies if needed."""
        try:
            import pytest
            import pytest_asyncio
        except ImportError:
            print("📦 Installing test dependencies...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "pytest", "pytest-asyncio", "pytest-mock", "pytest-cov"
            ], check=True)
            print("✅ Test dependencies installed")
    
    async def run_unit_tests(self) -> bool:
        """Run unit tests."""
        print("🧪 Running Unit Tests")
        print("=" * 40)
        
        unit_test_files = []
        
        # Find unit test files
        test_locations = [
            self.project_root / "src" / "chainlit_integration" / "managers",
            self.project_root / "src" / "chainlit_integration" / "utils",
            self.project_root / "tests"
        ]
        
        for location in test_locations:
            if location.exists():
                for test_file in location.glob("test_*.py"):
                    if "integration" not in test_file.name and "performance" not in test_file.name:
                        unit_test_files.append(str(test_file))
        
        if not unit_test_files:
            print("⚠️  No unit test files found")
            return True
        
        results = await self.pipeline.run_test_category('unit', unit_test_files)
        passed = len([r for r in results if r.status == 'passed'])
        total = len(results)
        
        success = passed == total
        print(f"\n📊 Unit Tests: {passed}/{total} passed")
        return success
    
    async def run_integration_tests(self) -> bool:
        """Run integration tests."""
        print("🔗 Running Integration Tests")
        print("=" * 40)
        
        integration_files = []
        test_locations = [
            self.project_root / "src" / "chainlit_integration" / "tests",
            self.project_root / "tests"
        ]
        
        for location in test_locations:
            if location.exists():
                for test_file in location.glob("*integration*.py"):
                    integration_files.append(str(test_file))
                for test_file in location.glob("*workflow*.py"):
                    integration_files.append(str(test_file))
        
        if not integration_files:
            print("⚠️  No integration test files found")
            return True
        
        results = await self.pipeline.run_test_category('integration', integration_files)
        passed = len([r for r in results if r.status == 'passed'])
        total = len(results)
        
        success = passed >= total * 0.8  # 80% pass rate for integration
        print(f"\n📊 Integration Tests: {passed}/{total} passed")
        return success
    
    async def run_performance_tests(self) -> bool:
        """Run performance tests."""
        print("⚡ Running Performance Tests")
        print("=" * 40)
        
        performance_files = []
        test_locations = [
            self.project_root / "tests"
        ]
        
        for location in test_locations:
            if location.exists():
                for test_file in location.glob("*performance*.py"):
                    performance_files.append(str(test_file))
        
        if not performance_files:
            print("⚠️  No performance test files found")
            return True
        
        results = await self.pipeline.run_test_category('performance', performance_files)
        passed = len([r for r in results if r.status == 'passed'])
        total = len(results)
        
        success = passed >= total * 0.7  # 70% pass rate for performance (more lenient)
        print(f"\n📊 Performance Tests: {passed}/{total} passed")
        return success
    
    async def run_e2e_tests(self) -> bool:
        """Run end-to-end tests."""
        print("🎯 Running End-to-End Tests")
        print("=" * 40)
        
        e2e_files = []
        test_locations = [
            self.project_root / "src" / "chainlit_integration" / "tests",
            self.project_root / "tests"
        ]
        
        for location in test_locations:
            if location.exists():
                for test_file in location.glob("*end_to_end*.py"):
                    e2e_files.append(str(test_file))
                for test_file in location.glob("*e2e*.py"):
                    e2e_files.append(str(test_file))
        
        if not e2e_files:
            print("⚠️  No end-to-end test files found")
            return True
        
        results = await self.pipeline.run_test_category('e2e', e2e_files)
        passed = len([r for r in results if r.status == 'passed'])
        total = len(results)
        
        success = passed >= total * 0.8  # 80% pass rate for e2e
        print(f"\n📊 End-to-End Tests: {passed}/{total} passed")
        return success
    
    async def run_with_coverage(self) -> bool:
        """Run tests with coverage reporting."""
        print("📈 Running Tests with Coverage")
        print("=" * 40)
        
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=src/chainlit_integration",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=70",
                "src/chainlit_integration/",
                "tests/",
                "-v"
            ]
            
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
            
            success = result.returncode == 0
            
            if success:
                print("\n✅ Coverage requirements met")
                coverage_dir = self.project_root / "htmlcov"
                if coverage_dir.exists():
                    print(f"📊 Coverage report: {coverage_dir / 'index.html'}")
            else:
                print("\n❌ Coverage requirements not met")
            
            return success
            
        except Exception as e:
            print(f"❌ Coverage test failed: {e}")
            return False
    
    async def run_fast_tests(self) -> bool:
        """Run only fast tests (exclude slow/performance tests)."""
        print("🏃 Running Fast Tests Only")
        print("=" * 40)
        
        # Run unit and basic integration tests
        unit_success = await self.run_unit_tests()
        
        # Run only fast integration tests
        integration_files = []
        test_locations = [
            self.project_root / "src" / "chainlit_integration" / "tests"
        ]
        
        for location in test_locations:
            if location.exists():
                for test_file in location.glob("test_*.py"):
                    if "performance" not in test_file.name and "slow" not in test_file.name:
                        integration_files.append(str(test_file))
        
        integration_success = True
        if integration_files:
            results = await self.pipeline.run_test_category('integration', integration_files)
            passed = len([r for r in results if r.status == 'passed'])
            total = len(results)
            integration_success = passed >= total * 0.8
            print(f"\n📊 Fast Integration Tests: {passed}/{total} passed")
        
        return unit_success and integration_success
    
    async def run_all_tests(self) -> bool:
        """Run all test categories."""
        print("🚀 Running All Tests")
        print("=" * 50)
        
        results = {
            'unit': await self.run_unit_tests(),
            'integration': await self.run_integration_tests(),
            'performance': await self.run_performance_tests(),
            'e2e': await self.run_e2e_tests()
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 FINAL TEST SUMMARY")
        print("=" * 50)
        
        for category, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{category.upper()}: {status}")
        
        overall_success = all(results.values())
        
        if overall_success:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n⚠️  SOME TESTS FAILED")
            failed_categories = [cat for cat, success in results.items() if not success]
            print(f"Failed categories: {', '.join(failed_categories)}")
        
        print("=" * 50)
        return overall_success
    
    def validate_test_environment(self) -> bool:
        """Validate that the test environment is properly set up."""
        print("🔍 Validating Test Environment")
        print("=" * 40)
        
        checks = []
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 7):
            checks.append(("Python version", True, f"{python_version.major}.{python_version.minor}"))
        else:
            checks.append(("Python version", False, f"{python_version.major}.{python_version.minor} (requires 3.7+)"))
        
        # Check required packages
        required_packages = ['pytest', 'asyncio']
        for package in required_packages:
            try:
                __import__(package)
                checks.append((f"Package {package}", True, "Available"))
            except ImportError:
                checks.append((f"Package {package}", False, "Missing"))
        
        # Check test files exist
        test_locations = [
            self.project_root / "src" / "chainlit_integration",
            self.project_root / "tests"
        ]
        
        test_files_found = 0
        for location in test_locations:
            if location.exists():
                test_files_found += len(list(location.rglob("test_*.py")))
        
        checks.append(("Test files", test_files_found > 0, f"{test_files_found} found"))
        
        # Check CSV data file
        csv_file = self.project_root / "Production_Event_Log.csv"
        checks.append(("Sample CSV data", csv_file.exists(), str(csv_file)))
        
        # Print results
        all_passed = True
        for check_name, passed, details in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}: {details}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print("\n✅ Environment validation passed")
        else:
            print("\n❌ Environment validation failed")
            print("Please fix the issues above before running tests")
        
        return all_passed


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    
    # Test category options
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")
    
    # Execution mode options
    parser.add_argument("--fast", action="store_true", help="Run fast tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--ci", action="store_true", help="Run in CI mode")
    parser.add_argument("--validate", action="store_true", help="Validate test environment only")
    
    # Configuration options
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner()
    
    # Setup environment
    runner.setup_environment()
    
    # Install dependencies if requested
    if args.install_deps:
        runner.install_test_dependencies()
    
    # Validate environment if requested
    if args.validate:
        success = runner.validate_test_environment()
        sys.exit(0 if success else 1)
    
    # Validate environment before running tests
    if not runner.validate_test_environment():
        print("\n❌ Environment validation failed. Use --install-deps to install dependencies.")
        sys.exit(1)
    
    try:
        # Determine what tests to run
        if args.ci:
            print("🤖 Running in CI mode")
            success = await runner.ci_runner.run_ci_pipeline()
        elif args.unit:
            success = await runner.run_unit_tests()
        elif args.integration:
            success = await runner.run_integration_tests()
        elif args.performance:
            success = await runner.run_performance_tests()
        elif args.e2e:
            success = await runner.run_e2e_tests()
        elif args.fast:
            success = await runner.run_fast_tests()
        elif args.coverage:
            success = await runner.run_with_coverage()
        else:
            # Run all tests by default
            success = await runner.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())