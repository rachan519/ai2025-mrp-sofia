# 🎓 Learning Plan Generator

An AI-powered application that creates personalized learning plans using LangChain AI 2, LangGraph, and Google Gemini. The system uses a multi-agent workflow to analyze knowledge gaps, plan learning paths, and provide detailed resources.

## ✨ Features

- **🤖 Multi-Agent AI System**: Four specialized AI agents work together to create comprehensive learning plans
- **🔍 Knowledge Gap Analysis**: Identifies what you need to learn based on your current background
- **📚 Structured Learning Paths**: Creates organized, step-by-step learning journeys
- **🎯 Personalized Resources**: Recommends specific resources, exercises, and assessment criteria
- **💾 Exportable Plans**: Save your learning plans as JSON files
- **🌐 Modern Web UI**: Beautiful Streamlit interface with progress tracking
- **📱 Terminal Interface**: Command-line version for automation and scripting

## 🏗️ Architecture

The system uses a LangGraph workflow with four specialized agents:

1. **Gap Analysis Agent**: Analyzes current knowledge and identifies learning gaps
2. **Topic Planning Agent**: Creates structured learning paths and objectives
3. **Topic Detail Agent**: Provides detailed breakdowns and specific resources
4. **Plan Combiner Agent**: Combines all components into a cohesive learning plan

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd skillbloom
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment**
   ```bash
   python setup_env.py
   ```
   Follow the prompts to enter your Google Gemini API key.

### Running the Application

#### 🌐 Web UI (Recommended)

Launch the modern Streamlit interface:

```bash
# Option 1: Use the launcher script
python run_app.py

# Option 2: Direct Streamlit command
streamlit run app.py
```

The web interface will open in your browser at `http://localhost:8501`

#### 📱 Terminal Interface

For command-line usage:

```bash
python main.py
```

#### 🧪 Test the System

Verify everything is working:

```bash
python test_system.py
```

## 🎯 Usage

### Web Interface

1. **Navigate to the "Generate Plan" page**
2. **Enter your learning topic** (e.g., "Machine Learning", "Python Programming")
3. **Describe your background** (e.g., "Beginner with no programming experience")
4. **Choose your preferred format** (Video, Text, or Audio)
5. **Click "Generate Learning Plan"**
6. **Review your personalized plan** with detailed breakdowns
7. **Download your plan** as a JSON file for future reference

### Terminal Interface

The terminal version provides the same functionality through interactive prompts:

- Enter your topic
- Describe your background
- Choose learning format
- Review the generated plan
- Option to save to file

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Model Settings

Adjust AI model parameters in `config.py`:

```python
MODEL_NAME = "gemini-2.5-pro"
TEMPERATURE = 0.7
MAX_TOKENS = 4000
```

## 📁 Project Structure

```
skillbloom/
├── app.py                 # Streamlit web application
├── main.py               # Terminal-based main application
├── run_app.py            # Streamlit launcher script
├── models.py             # Pydantic data models
├── agents.py             # AI agent implementations
├── workflow.py           # LangGraph workflow orchestration
├── config.py             # Configuration and environment
├── setup_env.py          # Environment setup helper
├── test_system.py        # System testing script
├── example_usage.py      # Example usage demonstrations
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## 🛠️ Technology Stack

- **Python 3.8+**: Core programming language
- **LangChain**: AI application framework
- **LangGraph**: Workflow orchestration
- **Google Gemini**: Advanced AI model
- **Streamlit**: Modern web interface
- **Pydantic**: Data validation and serialization
- **Python-dotenv**: Environment variable management

## 🔍 Understanding the Workflow

1. **Input Processing**: User provides topic, background, and learning preferences
2. **Gap Analysis**: AI analyzes current knowledge and identifies learning needs
3. **Topic Planning**: Creates structured learning objectives and timeline
4. **Detail Generation**: Provides specific resources, exercises, and assessments
5. **Plan Combination**: Integrates all components into a cohesive learning plan
6. **Output Generation**: Presents the plan in an organized, actionable format

## 📊 Example Output

The system generates comprehensive learning plans including:

- **Knowledge Gap Analysis**: Current vs. target skill levels
- **Learning Objectives**: Specific, measurable goals
- **Topic Breakdown**: Detailed subtopics and descriptions
- **Resource Recommendations**: Books, courses, videos, and exercises
- **Assessment Criteria**: How to measure progress and success
- **Timeline**: Estimated duration and milestones

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your `.env` file contains a valid `GOOGLE_API_KEY`
2. **Import Errors**: Make sure you're in the virtual environment and all dependencies are installed
3. **Port Conflicts**: If port 8501 is busy, Streamlit will automatically choose another port

### Getting Help

- Check the system test: `python test_system.py`
- Verify environment setup: `python setup_env.py check`
- Review error messages for specific guidance

## 🔮 Future Enhancements

- [ ] Progress tracking and learning analytics
- [ ] Integration with learning platforms (Coursera, Udemy, etc.)
- [ ] Collaborative learning plans
- [ ] Mobile app version
- [ ] Advanced customization options
- [ ] Learning path templates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the example usage files

---

**Happy Learning! 🎓✨** 