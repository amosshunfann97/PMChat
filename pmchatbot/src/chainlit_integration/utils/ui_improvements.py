"""
UI/UX improvements for Chainlit integration.

This module provides enhanced user interface elements and user experience
improvements for the process mining analysis tool.

Features:
- Enhanced message formatting
- Interactive elements
- User guidance and help
- Error message improvements
- Visual enhancements
- Accessibility improvements
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Conditional import for Chainlit
try:
    import chainlit as cl
except ImportError:
    # Mock for testing
    class MockChainlit:
        class Message:
            def __init__(self, content: str, actions=None):
                self.content = content
                self.actions = actions or []
            async def send(self): pass
        class Action:
            def __init__(self, name: str, value: str, label: str):
                self.name = name
                self.value = value
                self.label = label
        class Image:
            def __init__(self, path: str, name: str, display: str = "inline"):
                self.path = path
                self.name = name
                self.display = display
    cl = MockChainlit()


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages for consistent styling."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    HELP = "help"
    PROGRESS = "progress"


@dataclass
class UITheme:
    """UI theme configuration."""
    primary_color: str = "#2196F3"
    success_color: str = "#4CAF50"
    warning_color: str = "#FF9800"
    error_color: str = "#F44336"
    info_color: str = "#2196F3"
    help_color: str = "#9C27B0"


class MessageFormatter:
    """Enhanced message formatting for better UX."""
    
    def __init__(self, theme: UITheme = None):
        self.theme = theme or UITheme()
    
    def format_welcome_message(self) -> str:
        """Format the welcome message with enhanced styling."""
        return """# 🔍 Process Mining Analysis Tool

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;">
  <h2 style="margin: 0 0 10px 0;">Welcome to Process Mining Analysis!</h2>
  <p style="margin: 0; opacity: 0.9;">Analyze your process data through an intelligent chat interface</p>
</div>

## 🚀 Getting Started

This tool helps you discover insights from your process mining data through natural conversation. Here's how it works:

### Step 1: Choose Your AI Assistant
Select between **OpenAI GPT** (requires API key) or **Local Ollama** (runs locally)

### Step 2: Select Your Data
Choose which part/product you want to analyze from your process data

### Step 3: Ask Questions
Once processing is complete, ask questions about your process in natural language

---

**Ready to begin?** Choose your preferred AI model below:"""
    
    def format_success_message(self, title: str, content: str, next_steps: List[str] = None) -> str:
        """Format a success message."""
        message = f"""## ✅ {title}

<div style="background-color: #e8f5e8; border-left: 4px solid {self.theme.success_color}; padding: 15px; margin: 10px 0; border-radius: 5px;">
  {content}
</div>"""
        
        if next_steps:
            message += "\n\n**Next Steps:**\n"
            for i, step in enumerate(next_steps, 1):
                message += f"{i}. {step}\n"
        
        return message
    
    def format_error_message(self, title: str, error: str, suggestions: List[str] = None) -> str:
        """Format an error message with helpful suggestions."""
        message = f"""## ❌ {title}

<div style="background-color: #ffeaea; border-left: 4px solid {self.theme.error_color}; padding: 15px; margin: 10px 0; border-radius: 5px;">
  <strong>Error:</strong> {error}
</div>"""
        
        if suggestions:
            message += "\n\n**💡 Suggestions to fix this:**\n"
            for suggestion in suggestions:
                message += f"• {suggestion}\n"
        
        message += "\n\n**Need help?** Type `/help` for assistance or contact support."
        
        return message
    
    def format_warning_message(self, title: str, content: str, action_required: str = None) -> str:
        """Format a warning message."""
        message = f"""## ⚠️ {title}

<div style="background-color: #fff8e1; border-left: 4px solid {self.theme.warning_color}; padding: 15px; margin: 10px 0; border-radius: 5px;">
  {content}
</div>"""
        
        if action_required:
            message += f"\n\n**Action Required:** {action_required}"
        
        return message
    
    def format_info_message(self, title: str, content: str, details: Dict[str, Any] = None) -> str:
        """Format an informational message."""
        message = f"""## ℹ️ {title}

<div style="background-color: #e3f2fd; border-left: 4px solid {self.theme.info_color}; padding: 15px; margin: 10px 0; border-radius: 5px;">
  {content}
</div>"""
        
        if details:
            message += "\n\n**Details:**\n"
            for key, value in details.items():
                message += f"• **{key}**: {value}\n"
        
        return message
    
    def format_help_message(self, topic: str = "general") -> str:
        """Format help messages for different topics."""
        help_content = {
            "general": """## 🆘 Help & Support

### How to Use This Tool

1. **Select AI Model**: Choose between OpenAI or Ollama
2. **Choose Data**: Select which part/product to analyze
3. **Wait for Processing**: Data will be processed automatically
4. **Ask Questions**: Use natural language to explore your data

### Example Questions You Can Ask

**About Activities:**
• "What is the most common activity?"
• "Which activity takes the longest time?"
• "Show me the activity sequence"

**About Process Performance:**
• "What is the average process duration?"
• "Which cases take the longest?"
• "What are the bottlenecks?"

**About Variants:**
• "How many process variants exist?"
• "Which variant is most efficient?"
• "Compare different process paths"

### Commands
• `/help` - Show this help
• `/debug status` - Show system status (if debug mode enabled)
• Type "quit", "exit", "end", or "stop" to end session

### Troubleshooting
• If processing fails, try selecting a different part
• For API errors, check your OpenAI API key
• Refresh the page if you encounter issues""",
            
            "llm_selection": """## 🤖 AI Model Selection Help

### OpenAI GPT
**Pros:**
• Most advanced language model
• Best understanding of complex queries
• Excellent analysis capabilities

**Cons:**
• Requires API key (costs money)
• Needs internet connection
• Data sent to OpenAI servers

**Setup:** You'll need an OpenAI API key from https://platform.openai.com/

### Local Ollama
**Pros:**
• Completely private (runs locally)
• No API costs
• Works offline

**Cons:**
• Requires Ollama installation
• May be slower than OpenAI
• Less sophisticated responses

**Setup:** Install Ollama from https://ollama.ai/""",
            
            "data_processing": """## 📊 Data Processing Help

### What Happens During Processing?

1. **Data Loading**: Your CSV data is loaded and validated
2. **Filtering**: Data is filtered for the selected part
3. **Process Discovery**: PM4py analyzes the process flow
4. **Visualization**: Process diagrams are generated
5. **Indexing**: Data is prepared for intelligent querying

### Processing Time
• Small datasets (< 1000 cases): 30-60 seconds
• Medium datasets (1000-10000 cases): 1-3 minutes
• Large datasets (> 10000 cases): 3-10 minutes

### What You'll Get
• **Frequency DFG**: Shows how often activities occur
• **Performance DFG**: Shows time between activities
• **Query Interface**: Ask questions about your process"""
        }
        
        return help_content.get(topic, help_content["general"])
    
    def format_process_summary(self, part_name: str, stats: Dict[str, Any]) -> str:
        """Format process analysis summary."""
        return f"""## 📈 Process Analysis Summary for {part_name}

<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white; margin: 10px 0;">
  <h3 style="margin: 0 0 15px 0;">Analysis Complete!</h3>
  
  <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
    <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
      <div style="font-size: 24px; font-weight: bold;">{stats.get('total_cases', 'N/A')}</div>
      <div style="opacity: 0.9;">Total Cases</div>
    </div>
    
    <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
      <div style="font-size: 24px; font-weight: bold;">{stats.get('unique_activities', 'N/A')}</div>
      <div style="opacity: 0.9;">Unique Activities</div>
    </div>
    
    <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
      <div style="font-size: 24px; font-weight: bold;">{stats.get('avg_duration', 'N/A')}</div>
      <div style="opacity: 0.9;">Avg Duration</div>
    </div>
    
    <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 5px;">
      <div style="font-size: 24px; font-weight: bold;">{stats.get('process_variants', 'N/A')}</div>
      <div style="opacity: 0.9;">Process Variants</div>
    </div>
  </div>
</div>

**🎯 Ready for Analysis!** You can now ask questions about your process data. The visualizations above show your process flow and performance metrics."""


class InteractiveElements:
    """Create interactive UI elements."""
    
    @staticmethod
    def create_llm_selection_actions() -> List[cl.Action]:
        """Create LLM selection action buttons."""
        return [
            cl.Action(
                name="select_openai",
                value="openai",
                label="🤖 OpenAI GPT (Requires API Key)"
            ),
            cl.Action(
                name="select_ollama", 
                value="ollama",
                label="🏠 Local Ollama (Private & Free)"
            )
        ]
    
    @staticmethod
    def create_part_selection_actions(parts: List[str]) -> List[cl.Action]:
        """Create part selection action buttons."""
        actions = []
        
        # Limit to first 10 parts to avoid UI clutter
        display_parts = parts[:10]
        
        for part in display_parts:
            actions.append(cl.Action(
                name="part_selected",
                value=part,
                label=f"📦 {part}"
            ))
        
        # Add "Show More" if there are more parts
        if len(parts) > 10:
            actions.append(cl.Action(
                name="show_more_parts",
                value="show_more",
                label=f"📋 Show {len(parts) - 10} More Parts..."
            ))
        
        return actions
    
    @staticmethod
    def create_context_selection_actions() -> List[cl.Action]:
        """Create query context selection actions."""
        return [
            cl.Action(
                name="context_activity",
                value="activity", 
                label="🎯 Activity Analysis"
            ),
            cl.Action(
                name="context_process",
                value="process",
                label="🔄 Process Analysis"
            ),
            cl.Action(
                name="context_variant",
                value="variant",
                label="🌟 Variant Analysis"
            ),
            cl.Action(
                name="context_combined",
                value="combined",
                label="🔍 Combined Analysis"
            )
        ]
    
    @staticmethod
    def create_utility_actions() -> List[cl.Action]:
        """Create utility action buttons."""
        return [
            cl.Action(
                name="switch_part",
                value="switch",
                label="🔄 Switch Part"
            ),
            cl.Action(
                name="export_data",
                value="export",
                label="📊 Export Results"
            ),
            cl.Action(
                name="show_help",
                value="help",
                label="❓ Help"
            )
        ]


class AccessibilityHelper:
    """Improve accessibility of the interface."""
    
    @staticmethod
    def add_alt_text_to_images(images: List[cl.Image], descriptions: List[str]) -> List[cl.Image]:
        """Add alt text to images for accessibility."""
        for image, description in zip(images, descriptions):
            # In a real implementation, this would add proper alt text
            # For now, we'll add it to the name
            image.name = f"{image.name} - {description}"
        return images
    
    @staticmethod
    def format_for_screen_readers(content: str) -> str:
        """Format content to be more screen reader friendly."""
        # Replace emoji with text descriptions
        replacements = {
            "✅": "[Success]",
            "❌": "[Error]", 
            "⚠️": "[Warning]",
            "ℹ️": "[Information]",
            "🔍": "[Search]",
            "📊": "[Chart]",
            "🎯": "[Target]",
            "🔄": "[Process]",
            "⏳": "[Loading]",
            "🚀": "[Rocket]"
        }
        
        for emoji, text in replacements.items():
            content = content.replace(emoji, text)
        
        return content
    
    @staticmethod
    def create_keyboard_shortcuts_help() -> str:
        """Create keyboard shortcuts help."""
        return """## ⌨️ Keyboard Shortcuts

**Navigation:**
• Tab - Move between interactive elements
• Enter - Activate selected button
• Escape - Cancel current operation

**Quick Commands:**
• Type `/help` - Show help
• Type `/status` - Show system status
• Type `quit` - End session

**Accessibility:**
• Screen reader compatible
• High contrast mode supported
• Keyboard navigation enabled"""


class UserGuidance:
    """Provide contextual user guidance."""
    
    def __init__(self, formatter: MessageFormatter = None):
        self.formatter = formatter or MessageFormatter()
    
    async def show_onboarding_tips(self) -> None:
        """Show onboarding tips for new users."""
        tips = [
            "💡 **Tip 1**: Start by selecting your preferred AI model. OpenAI provides better responses but requires an API key.",
            "💡 **Tip 2**: Choose the part/product you want to analyze. This filters your data for focused analysis.",
            "💡 **Tip 3**: Wait for processing to complete. You'll see visualizations of your process flow.",
            "💡 **Tip 4**: Ask questions in natural language. Try 'What is the most common activity?' to start.",
            "💡 **Tip 5**: Use different contexts (Activity, Process, Variant) for different types of analysis."
        ]
        
        for tip in tips:
            await cl.Message(content=tip).send()
            # Small delay between tips
            import asyncio
            await asyncio.sleep(1)
    
    async def show_context_help(self, context: str) -> None:
        """Show help for specific contexts."""
        context_help = {
            "activity": """## 🎯 Activity Analysis Context

**What you can ask:**
• "What is the most frequent activity?"
• "Which activity takes the longest?"
• "Show me activities that come after X"
• "What resources perform activity Y?"

**Best for:** Understanding individual activities and their characteristics.""",
            
            "process": """## 🔄 Process Analysis Context

**What you can ask:**
• "What is the average process duration?"
• "How many cases follow the happy path?"
• "What are the process bottlenecks?"
• "Show me the process efficiency metrics"

**Best for:** Understanding overall process performance and flow.""",
            
            "variant": """## 🌟 Variant Analysis Context

**What you can ask:**
• "How many process variants exist?"
• "Which variant is most efficient?"
• "Compare variants by duration"
• "What causes process deviations?"

**Best for:** Understanding different process paths and their performance.""",
            
            "combined": """## 🔍 Combined Analysis Context

**What you can ask:**
• "Compare activity performance across variants"
• "How do resources affect process variants?"
• "What's the relationship between activities and outcomes?"
• "Analyze the complete process ecosystem"

**Best for:** Complex analysis combining multiple perspectives."""
        }
        
        help_content = context_help.get(context, "Context help not available.")
        await cl.Message(content=help_content).send()
    
    async def show_query_suggestions(self, context: str, part_name: str) -> None:
        """Show query suggestions based on context and part."""
        suggestions = {
            "activity": [
                f"What activities are involved in processing {part_name}?",
                f"Which activity takes the longest for {part_name}?",
                f"How often does quality control occur for {part_name}?",
                f"What is the sequence of activities for {part_name}?"
            ],
            "process": [
                f"What is the average processing time for {part_name}?",
                f"How efficient is the {part_name} process?",
                f"What are the bottlenecks in {part_name} processing?",
                f"How many {part_name} cases are processed per day?"
            ],
            "variant": [
                f"How many different ways can {part_name} be processed?",
                f"Which {part_name} process variant is fastest?",
                f"What causes variations in {part_name} processing?",
                f"Compare {part_name} process variants by cost"
            ]
        }
        
        context_suggestions = suggestions.get(context, [])
        if context_suggestions:
            content = f"## 💭 Suggested Questions for {part_name}\n\n"
            for i, suggestion in enumerate(context_suggestions, 1):
                content += f"{i}. {suggestion}\n"
            
            content += "\n*Click on any suggestion or type your own question!*"
            await cl.Message(content=content).send()


# Global instances
message_formatter = MessageFormatter()
interactive_elements = InteractiveElements()
accessibility_helper = AccessibilityHelper()
user_guidance = UserGuidance(message_formatter)