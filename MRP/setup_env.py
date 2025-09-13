#!/usr/bin/env python3
"""
Environment setup helper for Learning Plan Generator
"""

import os
import sys

def create_env_file():
    """Create a .env file with the required environment variables"""
    
    print("🔧 Learning Plan Generator - Environment Setup")
    print("=" * 50)
    
    # Check if .env file already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/n): ").lower().strip()
        if overwrite not in ['y', 'yes']:
            print("❌ Setup cancelled. Keeping existing .env file.")
            return
    
    print("\n📋 Setting up environment variables...")
    print("You'll need a Gemini API key from Google AI Studio.")
    print("Get it from: https://makersuite.google.com/app/apikey")
    
    # Get API key from user
    api_key = input("\n🔑 Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ API key cannot be empty. Setup cancelled.")
        return
    
    # Single iteration workflow - no need for iteration input
    max_iterations = 1
    
    # Create .env file content
    env_content = f"""# Learning Plan Generator Environment Variables
# Generated automatically by setup_env.py

# Gemini API Configuration
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY={api_key}

# Single iteration workflow

# Model Configuration (optional - these are defaults)
# MODEL_NAME=gemini-1.5-pro
# TEMPERATURE=0.7
# MAX_TOKENS=4000
"""
    
    # Write .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print(f"\n✅ .env file created successfully!")
        print(f"📁 File location: {os.path.abspath('.env')}")
        print(f"🔑 API key: {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) - 4 else '*' * len(api_key)}")
        print(f"🔄 Workflow: Single iteration")
        
        print("\n🚀 You're all set! Now you can:")
        print("   1. Run: python test_system.py (to test the system)")
        print("   2. Run: python main.py (interactive mode)")
        print("   3. Run: python example_usage.py (examples)")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return

def check_env_file():
    """Check if .env file exists and show its contents"""
    
    print("🔍 Checking environment configuration...")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Run: python setup_env.py to create it.")
        return False
    
    print("✅ .env file found!")
    
    try:
        with open('.env', 'r') as f:
            content = f.read()
        
        print("\n📋 Current environment configuration:")
        print("-" * 40)
        
        for line in content.split('\n'):
            if line.strip() and not line.startswith('#'):
                if 'GOOGLE_API_KEY' in line:
                    key, value = line.split('=', 1)
                    masked_value = '*' * (len(value) - 4) + value[-4:] if len(value) > 4 else '*' * len(value)
                    print(f"{key}={masked_value}")
                else:
                    print(line)
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

def main():
    """Main function"""
    
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        check_env_file()
    else:
        create_env_file()

if __name__ == "__main__":
    main() 