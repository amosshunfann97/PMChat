"""
Automated testing pipeline for continuous integration.

This module provides automated test execution, reporting, and validation
for the Chainlit integration system.

Features:
- Automated test discovery and execution
- Performance benchmarking
- Test result reporting
- Continuous integration support
- Test data validation
"""

import pytest
import asyncio
import json
import time
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    status: str  # 'passed', 'failed', 'skipped'
    duration: float
    error_message: Optional[str] = None
    category: str = 'unit'  # 'unit', 'integration', 'e2e', 'performance'


@dataclass
class TestSuiteReport:
    """Test suite report data structure."""
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    total_duration: float
    coverage_percentage: float
    performance_metrics: Dict[str, Any]
    test_results: List[TestResult]
    environment_info: Dict[str, str]


class TestAutomationPipeline:
    """Automated testing pipeline."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.test_results = []
        self.performance_metrics = {}
        self.start_time = None
        
    def discover_tests(self) -> List[str]:
        """Discover all test files in the project."""
        test_files = []
        
        # Discover test files in different locations
        test_locations = [
            self.project_root / "tests",
            self.project_root / "src" / "chainlit_integration" / "tests",
            self.project_root / "src" / "chainlit_integration" / "managers"
        ]
        
        for location in test_locations:
            if location.exists():
                for test_file in location.rglob("test_*.py"):
                    test_files.append(str(test_file))
        
        return test_files
    
    def categorize_tests(self, test_files: List[str]) -> Dict[str, List[str]]:
        """Categorize tests by type."""
        categories = {
            'unit': [],
            'integration': [],
            'e2e': [],
            'performance': []
        }
        
        for test_file in test_files:
            file_name = Path(test_file).name
            
            if 'performance' in file_name:
                categories['performance'].append(test_file)
            elif 'integration' in file_name or 'end_to_end' in file_name:
                categories['integration'].append(test_file)
            elif 'e2e' in file_name or 'workflow' in file_name:
                categories['e2e'].append(test_file)
            else:
                categories['unit'].append(test_file)
        
        return categories
    
    async def run_test_category(self, category: str, test_files: List[str]) -> List[TestResult]:
        """Run tests for a specific category."""
        results = []
        
        if not test_files:
            return results
        
        print(f"\n🧪 Running {category} tests...")
        print("=" * 50)
        
        for test_file in test_files:
            try:
                start_time = time.time()
                
                # Run pytest on the specific file
                cmd = [
                    sys.executable, "-m", "pytest", 
                    test_file, 
                    "-v", 
                    "--tb=short",
                    "--json-report",
                    f"--json-report-file={self.project_root}/test_report_{category}.json"
                ]
                
                if category == 'performance':
                    cmd.extend(["-m", "performance"])
                elif category == 'integration':
                    cmd.extend(["-m", "integration"])
                elif category == 'e2e':
                    cmd.extend(["-m", "e2e"])
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                duration = time.time() - start_time
                
                # Parse result
                if result.returncode == 0:
                    status = 'passed'
                    error_message = None
                else:
                    status = 'failed'
                    error_message = result.stderr or result.stdout
                
                test_result = TestResult(
                    test_name=Path(test_file).name,
                    status=status,
                    duration=duration,
                    error_message=error_message,
                    category=category
                )
                
                results.append(test_result)
                
                # Print result
                status_emoji = "✅" if status == 'passed' else "❌"
                print(f"  {status_emoji} {test_result.test_name} ({duration:.2f}s)")
                
                if error_message and len(error_message) < 200:
                    print(f"     Error: {error_message}")
                
            except Exception as e:
                error_result = TestResult(
                    test_name=Path(test_file).name,
                    status='failed',
                    duration=0.0,
                    error_message=str(e),
                    category=category
                )
                results.append(error_result)
                print(f"  ❌ {error_result.test_name} (Error: {str(e)})")
        
        return results
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from test runs."""
        metrics = {
            'test_execution_time': sum(r.duration for r in self.test_results),
            'average_test_time': 0.0,
            'slowest_tests': [],
            'failed_tests': [],
            'category_performance': {}
        }
        
        if self.test_results:
            metrics['average_test_time'] = metrics['test_execution_time'] / len(self.test_results)
            
            # Find slowest tests
            sorted_tests = sorted(self.test_results, key=lambda x: x.duration, reverse=True)
            metrics['slowest_tests'] = [
                {'name': t.test_name, 'duration': t.duration, 'category': t.category}
                for t in sorted_tests[:5]
            ]
            
            # Find failed tests
            metrics['failed_tests'] = [
                {'name': t.test_name, 'error': t.error_message, 'category': t.category}
                for t in self.test_results if t.status == 'failed'
            ]
            
            # Category performance
            categories = set(t.category for t in self.test_results)
            for category in categories:
                category_tests = [t for t in self.test_results if t.category == category]
                metrics['category_performance'][category] = {
                    'total_tests': len(category_tests),
                    'passed': len([t for t in category_tests if t.status == 'passed']),
                    'failed': len([t for t in category_tests if t.status == 'failed']),
                    'total_duration': sum(t.duration for t in category_tests),
                    'average_duration': sum(t.duration for t in category_tests) / len(category_tests) if category_tests else 0
                }
        
        return metrics
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get environment information."""
        import platform
        
        env_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add package versions if available
        try:
            import pkg_resources
            packages = ['pytest', 'asyncio', 'pandas', 'chainlit']
            for package in packages:
                try:
                    version = pkg_resources.get_distribution(package).version
                    env_info[f'{package}_version'] = version
                except:
                    env_info[f'{package}_version'] = 'unknown'
        except ImportError:
            pass
        
        return env_info
    
    def calculate_coverage(self) -> float:
        """Calculate test coverage percentage (simplified)."""
        # This is a simplified coverage calculation
        # In a real scenario, you'd use coverage.py or similar tools
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t.status == 'passed'])
        
        if total_tests == 0:
            return 0.0
        
        return (passed_tests / total_tests) * 100
    
    def generate_report(self) -> TestSuiteReport:
        """Generate comprehensive test report."""
        total_tests = len(self.test_results)
        passed = len([t for t in self.test_results if t.status == 'passed'])
        failed = len([t for t in self.test_results if t.status == 'failed'])
        skipped = len([t for t in self.test_results if t.status == 'skipped'])
        
        total_duration = sum(t.duration for t in self.test_results)
        coverage = self.calculate_coverage()
        performance_metrics = self.collect_performance_metrics()
        environment_info = self.get_environment_info()
        
        report = TestSuiteReport(
            timestamp=datetime.now().isoformat(),
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            total_duration=total_duration,
            coverage_percentage=coverage,
            performance_metrics=performance_metrics,
            test_results=self.test_results,
            environment_info=environment_info
        )
        
        return report
    
    def save_report(self, report: TestSuiteReport, filename: str = None):
        """Save test report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{timestamp}.json"
        
        report_path = self.project_root / filename
        
        # Convert to dict for JSON serialization
        report_dict = asdict(report)
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\n📊 Test report saved to: {report_path}")
        return report_path
    
    def print_summary(self, report: TestSuiteReport):
        """Print test summary to console."""
        print("\n" + "=" * 60)
        print("🎯 TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        print(f"📅 Timestamp: {report.timestamp}")
        print(f"⏱️  Total Duration: {report.total_duration:.2f}s")
        print(f"🧪 Total Tests: {report.total_tests}")
        print(f"✅ Passed: {report.passed}")
        print(f"❌ Failed: {report.failed}")
        print(f"⏭️  Skipped: {report.skipped}")
        print(f"📈 Coverage: {report.coverage_percentage:.1f}%")
        
        # Category breakdown
        print("\n📊 Category Breakdown:")
        for category, metrics in report.performance_metrics['category_performance'].items():
            print(f"  {category.upper()}: {metrics['passed']}/{metrics['total_tests']} passed ({metrics['total_duration']:.2f}s)")
        
        # Performance highlights
        if report.performance_metrics['slowest_tests']:
            print("\n🐌 Slowest Tests:")
            for test in report.performance_metrics['slowest_tests'][:3]:
                print(f"  • {test['name']}: {test['duration']:.2f}s ({test['category']})")
        
        # Failed tests
        if report.performance_metrics['failed_tests']:
            print("\n💥 Failed Tests:")
            for test in report.performance_metrics['failed_tests'][:3]:
                error_preview = test['error'][:100] + "..." if len(test['error']) > 100 else test['error']
                print(f"  • {test['name']}: {error_preview}")
        
        # Overall status
        success_rate = (report.passed / report.total_tests * 100) if report.total_tests > 0 else 0
        if success_rate >= 90:
            print(f"\n🎉 EXCELLENT: {success_rate:.1f}% success rate!")
        elif success_rate >= 75:
            print(f"\n👍 GOOD: {success_rate:.1f}% success rate")
        elif success_rate >= 50:
            print(f"\n⚠️  NEEDS IMPROVEMENT: {success_rate:.1f}% success rate")
        else:
            print(f"\n🚨 CRITICAL: {success_rate:.1f}% success rate - immediate attention required!")
        
        print("=" * 60)
    
    async def run_full_pipeline(self, categories: List[str] = None) -> TestSuiteReport:
        """Run the complete testing pipeline."""
        self.start_time = time.time()
        self.test_results = []
        
        print("🚀 Starting Automated Test Pipeline")
        print("=" * 60)
        
        # Discover tests
        test_files = self.discover_tests()
        categorized_tests = self.categorize_tests(test_files)
        
        print(f"📁 Discovered {len(test_files)} test files")
        for category, files in categorized_tests.items():
            if files:
                print(f"  • {category}: {len(files)} files")
        
        # Run tests by category
        if categories is None:
            categories = ['unit', 'integration', 'performance', 'e2e']
        
        for category in categories:
            if category in categorized_tests and categorized_tests[category]:
                results = await self.run_test_category(category, categorized_tests[category])
                self.test_results.extend(results)
        
        # Generate and save report
        report = self.generate_report()
        report_path = self.save_report(report)
        
        # Print summary
        self.print_summary(report)
        
        return report


class ContinuousIntegrationRunner:
    """CI/CD specific test runner."""
    
    def __init__(self, project_root: str = None):
        self.pipeline = TestAutomationPipeline(project_root)
    
    async def run_ci_pipeline(self) -> bool:
        """Run CI pipeline and return success status."""
        try:
            report = await self.pipeline.run_full_pipeline(['unit', 'integration'])
            
            # CI success criteria
            success_rate = (report.passed / report.total_tests * 100) if report.total_tests > 0 else 0
            max_duration = 300  # 5 minutes max for CI
            
            ci_success = (
                success_rate >= 80 and  # At least 80% tests pass
                report.total_duration <= max_duration and  # Within time limit
                report.failed == 0  # No critical failures
            )
            
            if ci_success:
                print("\n✅ CI Pipeline: SUCCESS")
            else:
                print("\n❌ CI Pipeline: FAILED")
                print(f"   Success rate: {success_rate:.1f}% (required: 80%)")
                print(f"   Duration: {report.total_duration:.1f}s (max: {max_duration}s)")
                print(f"   Failed tests: {report.failed}")
            
            return ci_success
            
        except Exception as e:
            print(f"\n💥 CI Pipeline Error: {e}")
            return False
    
    async def run_nightly_pipeline(self) -> bool:
        """Run comprehensive nightly pipeline."""
        try:
            report = await self.pipeline.run_full_pipeline()
            
            # Nightly success criteria (more lenient)
            success_rate = (report.passed / report.total_tests * 100) if report.total_tests > 0 else 0
            
            nightly_success = success_rate >= 70  # At least 70% for nightly
            
            if nightly_success:
                print("\n🌙 Nightly Pipeline: SUCCESS")
            else:
                print("\n🌙 Nightly Pipeline: FAILED")
                print(f"   Success rate: {success_rate:.1f}% (required: 70%)")
            
            return nightly_success
            
        except Exception as e:
            print(f"\n💥 Nightly Pipeline Error: {e}")
            return False


# CLI interface
async def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Test Pipeline")
    parser.add_argument("--mode", choices=['full', 'ci', 'nightly'], default='full',
                       help="Test execution mode")
    parser.add_argument("--categories", nargs='+', 
                       choices=['unit', 'integration', 'performance', 'e2e'],
                       help="Test categories to run")
    parser.add_argument("--project-root", help="Project root directory")
    
    args = parser.parse_args()
    
    if args.mode == 'ci':
        runner = ContinuousIntegrationRunner(args.project_root)
        success = await runner.run_ci_pipeline()
        sys.exit(0 if success else 1)
    elif args.mode == 'nightly':
        runner = ContinuousIntegrationRunner(args.project_root)
        success = await runner.run_nightly_pipeline()
        sys.exit(0 if success else 1)
    else:
        pipeline = TestAutomationPipeline(args.project_root)
        report = await pipeline.run_full_pipeline(args.categories)
        success_rate = (report.passed / report.total_tests * 100) if report.total_tests > 0 else 0
        sys.exit(0 if success_rate >= 80 else 1)


if __name__ == "__main__":
    asyncio.run(main())