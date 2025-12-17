#!/usr/bin/env python
"""
IT Risk Classification API Test Script
Tests various API endpoints with different risk profiles
"""

import requests
import json
import time
from typing import Dict, Any

# API Base URL
API_URL = "http://localhost:8000"

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_response(title: str, data: Any):
    """Pretty print a response"""
    print(f"\n{title}:")
    print(json.dumps(data, indent=2, default=str))

def test_health_check():
    """Test health check endpoint"""
    print_section("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{API_URL}/health")
        print_response("Health Status", response.json())
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print_section("TEST 2: Model Information")
    
    try:
        response = requests.get(f"{API_URL}/model-info")
        data = response.json()
        print_response("Model Info", data)
        print(f"\nNumber of features: {len(data.get('features', []))}")
        print(f"Risk classes: {', '.join(data.get('risk_classes', []))}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_well_secured_infrastructure():
    """Test assessment for well-secured infrastructure"""
    print_section("TEST 3: Well-Secured Infrastructure Assessment")
    
    payload = {
        "firewall_enabled": 1,
        "encryption_level": 3,  # Strong
        "mfa_enabled": 1,
        "compliance_score": 95,
        "num_security_patches": 45,
        "unpatched_vulnerabilities": 2,
        "backup_enabled": 1,
        "disk_encryption": 1,
        "failed_login_attempts": 10,
        "unusual_traffic": 0,
        "privileged_access_changes": 2,
        "security_events_per_day": 50
    }
    
    try:
        response = requests.post(f"{API_URL}/assess", json=payload)
        data = response.json()
        print_response("Assessment Result", data)
        
        risk_class = data.get('risk_classification')
        confidence = data.get('confidence', 0)
        print(f"\n✓ Risk Level: {risk_class}")
        print(f"✓ Confidence: {confidence:.2%}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_poorly_secured_infrastructure():
    """Test assessment for poorly-secured infrastructure"""
    print_section("TEST 4: Poorly-Secured Infrastructure Assessment")
    
    payload = {
        "firewall_enabled": 0,
        "encryption_level": 0,  # None
        "mfa_enabled": 0,
        "compliance_score": 20,
        "num_security_patches": 5,
        "unpatched_vulnerabilities": 85,
        "backup_enabled": 0,
        "disk_encryption": 0,
        "failed_login_attempts": 500,
        "unusual_traffic": 1,
        "privileged_access_changes": 40,
        "security_events_per_day": 450
    }
    
    try:
        response = requests.post(f"{API_URL}/assess", json=payload)
        data = response.json()
        print_response("Assessment Result", data)
        
        risk_class = data.get('risk_classification')
        confidence = data.get('confidence', 0)
        print(f"\n✓ Risk Level: {risk_class}")
        print(f"✓ Confidence: {confidence:.2%}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_medium_risk_infrastructure():
    """Test assessment for medium-risk infrastructure"""
    print_section("TEST 5: Medium-Risk Infrastructure Assessment")
    
    payload = {
        "firewall_enabled": 1,
        "encryption_level": 1,  # Basic
        "mfa_enabled": 0,
        "compliance_score": 55,
        "num_security_patches": 15,
        "unpatched_vulnerabilities": 25,
        "backup_enabled": 1,
        "disk_encryption": 0,
        "failed_login_attempts": 150,
        "unusual_traffic": 0,
        "privileged_access_changes": 10,
        "security_events_per_day": 150
    }
    
    try:
        response = requests.post(f"{API_URL}/assess", json=payload)
        data = response.json()
        print_response("Assessment Result", data)
        
        risk_class = data.get('risk_classification')
        confidence = data.get('confidence', 0)
        print(f"\n✓ Risk Level: {risk_class}")
        print(f"✓ Confidence: {confidence:.2%}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_assessment():
    """Test batch assessment endpoint"""
    print_section("TEST 6: Batch Assessment (3 Infrastructures)")
    
    payload = {
        "assessments": [
            {
                "firewall_enabled": 1, "encryption_level": 3, "mfa_enabled": 1, "compliance_score": 95,
                "num_security_patches": 45, "unpatched_vulnerabilities": 2, "backup_enabled": 1, 
                "disk_encryption": 1, "failed_login_attempts": 10, "unusual_traffic": 0, 
                "privileged_access_changes": 2, "security_events_per_day": 50
            },
            {
                "firewall_enabled": 0, "encryption_level": 0, "mfa_enabled": 0, "compliance_score": 20,
                "num_security_patches": 5, "unpatched_vulnerabilities": 85, "backup_enabled": 0, 
                "disk_encryption": 0, "failed_login_attempts": 500, "unusual_traffic": 1, 
                "privileged_access_changes": 40, "security_events_per_day": 450
            },
            {
                "firewall_enabled": 1, "encryption_level": 2, "mfa_enabled": 1, "compliance_score": 75,
                "num_security_patches": 30, "unpatched_vulnerabilities": 10, "backup_enabled": 1, 
                "disk_encryption": 1, "failed_login_attempts": 50, "unusual_traffic": 0, 
                "privileged_access_changes": 5, "security_events_per_day": 100
            }
        ]
    }
    
    try:
        response = requests.post(f"{API_URL}/assess-batch", json=payload)
        data = response.json()
        
        print(f"\nBatch Assessment Results ({data.get('total')} infrastructures):\n")
        
        for idx, assessment in enumerate(data.get('assessments', []), 1):
            risk_class = assessment.get('risk_classification')
            confidence = assessment.get('confidence', 0)
            probs = assessment.get('risk_probabilities', {})
            
            print(f"Infrastructure {idx}:")
            print(f"  Risk Level: {risk_class}")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Probabilities:")
            for risk, prob in probs.items():
                print(f"    - {risk}: {prob:.2%}")
            print()
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "IT RISK CLASSIFICATION API TEST SUITE" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    
    print(f"\nAPI URL: {API_URL}")
    print("Note: Make sure the API server is running (python scripts/it_risk_api.py)")
    
    # Wait a moment for API to be ready
    time.sleep(1)
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Well-Secured Infrastructure", test_well_secured_infrastructure),
        ("Poorly-Secured Infrastructure", test_poorly_secured_infrastructure),
        ("Medium-Risk Infrastructure", test_medium_risk_infrastructure),
        ("Batch Assessment", test_batch_assessment),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_all_tests()
