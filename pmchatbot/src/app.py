import chainlit as cl
import pandas as pd
import io
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@cl.on_chat_start
async def start():
    await cl.Message(
        content="Welcome! Please upload a CSV file to get started. I can help you analyze and answer questions about your data."
    ).send()

async def auto_detect_columns(df):
    """Attempt to auto-detect columns and ask for confirmation"""
    columns = df.columns.tolist()
    
    # Store columns in session
    cl.user_session.set("available_columns", columns)
    
    # Simple heuristics for auto-detection
    case_id_candidates = [col for col in columns if any(term in col.lower() for term in ['case', 'id', 'instance', 'trace', 'process'])]
    activity_candidates = [col for col in columns if any(term in col.lower() for term in ['activity', 'event', 'task', 'step', 'action', 'name'])]
    timestamp_candidates = [col for col in columns if any(term in col.lower() for term in ['time', 'date', 'timestamp', 'start', 'end', 'created', 'occurred'])]
    
    # Also check data types for better detection
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    datetime_cols = []
    
    # Try to detect datetime columns
    for col in columns:
        try:
            sample_data = df[col].dropna().head(50)  # Smaller sample
            if len(sample_data) == 0:
                continue
                
            # Check if column name suggests datetime
            col_lower = col.lower()
            if any(term in col_lower for term in ['time', 'date', 'timestamp', 'created', 'updated']):
                # Try to parse a few samples
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(sample_data.iloc[:5], errors='coerce')
                        if parsed.notna().sum() >= 3:  # At least 3 out of 5 parsed successfully
                            datetime_cols.append(col)
                except:
                    pass
        except:
            pass
    
    # Smart selection logic
    suggestions = {}
    
    # Case ID: prefer columns with 'case' or 'id' in name, or first numeric column
    if case_id_candidates:
        suggestions['case_id'] = case_id_candidates[0]
    elif numeric_cols:
        suggestions['case_id'] = numeric_cols[0]
    else:
        suggestions['case_id'] = columns[0]
    
    # Activity: prefer columns with activity-related terms
    if activity_candidates:
        suggestions['activity'] = activity_candidates[0]
    else:
        # Find a text column that's not the case_id
        text_cols = [col for col in columns if col != suggestions['case_id'] and df[col].dtype == 'object']
        suggestions['activity'] = text_cols[0] if text_cols else columns[1] if len(columns) > 1 else columns[0]
    
    # Timestamp: prefer datetime columns or columns with time-related terms
    if datetime_cols:
        suggestions['timestamp'] = datetime_cols[0]
    elif timestamp_candidates:
        suggestions['timestamp'] = timestamp_candidates[0]
    else:
        # Use last column as fallback
        suggestions['timestamp'] = columns[-1]
    
    # Calculate confidence score
    confidence_score = 0
    if case_id_candidates: confidence_score += 30
    if activity_candidates: confidence_score += 30
    if datetime_cols or timestamp_candidates: confidence_score += 40
    
    confidence_level = "High" if confidence_score >= 70 else "Medium" if confidence_score >= 40 else "Low"
    
    # Show sample data with detected columns
    sample_data = df[[suggestions['case_id'], suggestions['activity'], suggestions['timestamp']]].head(3)
    
    message = f"""**🤖 Auto-Detection Results**

**Confidence Level:** {confidence_level} ({confidence_score}%)

Based on column names and data types, I suggest:
- **Case ID:** `{suggestions['case_id']}`
- **Activity:** `{suggestions['activity']}`
- **Timestamp:** `{suggestions['timestamp']}`

**Sample data with detected columns:**
```
{sample_data.to_string()}
```

**Available columns:** {', '.join(columns)}"""
    
    # Create action buttons for accept/reject
    accept_action = cl.Action(
        name="accept_auto_detection",
        value="accept",
        label="✅ Accept These Suggestions",
        payload={"action": "accept", "suggestions": suggestions}
    )
    
    reject_action = cl.Action(
        name="reject_auto_detection", 
        value="reject",
        label="❌ Select Manually",
        payload={"action": "reject"}
    )
    
    cl.user_session.set("column_suggestions", suggestions)
    
    await cl.Message(
        content=message,
        actions=[accept_action, reject_action]
    ).send()

async def show_column_selectors(df):
    """Show column selectors for case_id, activity, and timestamp using action buttons"""
    columns = df.columns.tolist()
    
    # Store columns in session for reference
    cl.user_session.set("available_columns", columns)
    cl.user_session.set("selection_step", "case_id")
    cl.user_session.set("selected_columns", [])  # Track selected columns
    
    # Create action buttons for case_id column selection
    actions = []
    for i, col in enumerate(columns[:10]):  # Limit to first 10 columns to avoid too many buttons
        actions.append(cl.Action(
            name=f"select_case_id_{i}",
            value=col,
            label=f"📋 {col}",
            payload={"column": col, "type": "case_id"}
        ))
    
    message_content = f"""**Step 1/3: Select Case ID Column**

Please select which column contains the **Case ID** (unique identifier for each process instance):

**Available columns:** {', '.join(columns)}

Click on the appropriate column below:"""
    
    await cl.Message(
        content=message_content,
        actions=actions
    ).send()

# Auto-detection action callbacks
@cl.action_callback("accept_auto_detection")
async def on_accept_auto_detection(action):
    """Handle accepting auto-detected columns"""
    suggestions = action.payload["suggestions"]
    
    # Store column mappings
    column_mapping = {
        'case_id': suggestions['case_id'],
        'activity': suggestions['activity'],
        'timestamp': suggestions['timestamp']
    }
    cl.user_session.set("column_mapping", column_mapping)
    
    # Get dataframe and show confirmation
    df = cl.user_session.get("dataframe")
    sample_data = df[[suggestions['case_id'], suggestions['activity'], suggestions['timestamp']]].head()
    
    confirmation_msg = f"""**✅ Column Selection Complete!**

**Final Column Mapping:**
- **Case ID:** `{suggestions['case_id']}`
- **Activity:** `{suggestions['activity']}`
- **Timestamp:** `{suggestions['timestamp']}`

📋 **Sample data with selected columns:**
```
{sample_data.to_string()}
```

🎉 **Setup Complete!** Now you can perform process mining analysis! Try asking:
- 'Show process statistics'
- 'Find most common activities'
- 'Show case durations'"""
    
    await cl.Message(content=confirmation_msg).send()

@cl.action_callback("reject_auto_detection")
async def on_reject_auto_detection(action):
    """Handle rejecting auto-detected columns and switch to manual selection"""
    df = cl.user_session.get("dataframe")
    await show_column_selectors(df)

# Manual selection action callbacks - Case ID
@cl.action_callback("select_case_id_0")
@cl.action_callback("select_case_id_1") 
@cl.action_callback("select_case_id_2")
@cl.action_callback("select_case_id_3")
@cl.action_callback("select_case_id_4")
@cl.action_callback("select_case_id_5")
@cl.action_callback("select_case_id_6")
@cl.action_callback("select_case_id_7")
@cl.action_callback("select_case_id_8")
@cl.action_callback("select_case_id_9")
async def on_case_id_selected(action):
    """Handle case ID column selection"""
    case_id_col = action.payload["column"]
    cl.user_session.set("selected_case_id", case_id_col)
    
    # Add to selected columns list
    selected_columns = cl.user_session.get("selected_columns", [])
    selected_columns.append(case_id_col)
    cl.user_session.set("selected_columns", selected_columns)
    
    # Move to activity column selection
    all_columns = cl.user_session.get("available_columns")
    # Filter out already selected columns
    remaining_columns = [col for col in all_columns if col not in selected_columns]
    
    actions = []
    for i, col in enumerate(remaining_columns[:10]):  # Show only remaining columns
        actions.append(cl.Action(
            name=f"select_activity_{i}",
            value=col,
            label=f"⚡ {col}",
            payload={"column": col, "type": "activity"}
        ))
    
    message_content = f"""**Step 2/3: Select Activity Column**

✅ **Case ID Column:** {case_id_col}

Please select which column contains the **Activity** names:

**Remaining columns:** {', '.join(remaining_columns)}

Click on the appropriate column below:"""
    
    await cl.Message(
        content=message_content,
        actions=actions
    ).send()

# Manual selection action callbacks - Activity
@cl.action_callback("select_activity_0")
@cl.action_callback("select_activity_1")
@cl.action_callback("select_activity_2")
@cl.action_callback("select_activity_3")
@cl.action_callback("select_activity_4")
@cl.action_callback("select_activity_5")
@cl.action_callback("select_activity_6")
@cl.action_callback("select_activity_7")
@cl.action_callback("select_activity_8")
@cl.action_callback("select_activity_9")
async def on_activity_selected(action):
    """Handle activity column selection"""
    activity_col = action.payload["column"]
    cl.user_session.set("selected_activity", activity_col)
    
    # Add to selected columns list
    selected_columns = cl.user_session.get("selected_columns", [])
    selected_columns.append(activity_col)
    cl.user_session.set("selected_columns", selected_columns)
    
    # Move to timestamp column selection
    all_columns = cl.user_session.get("available_columns")
    # Filter out already selected columns
    remaining_columns = [col for col in all_columns if col not in selected_columns]
    
    actions = []
    for i, col in enumerate(remaining_columns[:10]):  # Show only remaining columns
        actions.append(cl.Action(
            name=f"select_timestamp_{i}",
            value=col,
            label=f"🕒 {col}",
            payload={"column": col, "type": "timestamp"}
        ))
    
    case_id_col = cl.user_session.get("selected_case_id")
    message_content = f"""**Step 3/3: Select Timestamp Column**

✅ **Case ID Column:** {case_id_col}
✅ **Activity Column:** {activity_col}

Please select which column contains the **Timestamp** information:

**Remaining columns:** {', '.join(remaining_columns)}

Click on the appropriate column below:"""
    
    await cl.Message(
        content=message_content,
        actions=actions
    ).send()

# Manual selection action callbacks - Timestamp
@cl.action_callback("select_timestamp_0")
@cl.action_callback("select_timestamp_1")
@cl.action_callback("select_timestamp_2")
@cl.action_callback("select_timestamp_3")
@cl.action_callback("select_timestamp_4")
@cl.action_callback("select_timestamp_5")
@cl.action_callback("select_timestamp_6")
@cl.action_callback("select_timestamp_7")
@cl.action_callback("select_timestamp_8")
@cl.action_callback("select_timestamp_9")
async def on_timestamp_selected(action):
    """Handle timestamp column selection and complete setup"""
    timestamp_col = action.payload["column"]
    cl.user_session.set("selected_timestamp", timestamp_col)
    
    # Get all selected columns
    case_id_col = cl.user_session.get("selected_case_id")
    activity_col = cl.user_session.get("selected_activity")
    
    # Store column mappings
    column_mapping = {
        'case_id': case_id_col,
        'activity': activity_col,
        'timestamp': timestamp_col
    }
    cl.user_session.set("column_mapping", column_mapping)
    
    # Clear selection tracking
    cl.user_session.set("selected_columns", [])
    cl.user_session.set("selection_step", None)
    
    # Get dataframe and show confirmation
    df = cl.user_session.get("dataframe")
    
    confirmation_msg = f"""**✅ Column Selection Complete!**

**Final Column Mapping:**
- **Case ID:** {case_id_col}
- **Activity:** {activity_col}
- **Timestamp:** {timestamp_col}

📋 **Sample data with selected columns:**
"""
    
    await cl.Message(content=confirmation_msg).send()
    
    # Show sample data with selected columns
    sample_data = df[[case_id_col, activity_col, timestamp_col]].head()
    await cl.Message(
        content=f"```\n{sample_data.to_string()}\n```"
    ).send()
    
    await cl.Message(
        content="🎉 **Setup Complete!** Now you can perform process mining analysis! Try asking:\n- 'Show process statistics'\n- 'Find most common activities'\n- 'Show case durations'"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Check if there are any file attachments
    if message.elements:
        for element in message.elements:
            if element.name.endswith('.csv'):
                # Process CSV file
                try:
                    # Check if element has content or path
                    if hasattr(element, 'content') and element.content:
                        # Use content directly with enhanced processing
                        import tempfile
                        
                        # Save content to temporary file for processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            tmp_file.write(element.content)
                            tmp_file_path = tmp_file.name
                        
                        try:
                            from csv_processor import CsvProcessor
                            processor = CsvProcessor()
                            result = processor.read_csv_with_options(tmp_file_path)
                            
                            if result['success']:
                                df = result['data']
                                delimiter = result['delimiter']
                            else:
                                await cl.Message(content=f"Error processing CSV: {result['error']}").send()
                                continue
                        except ImportError:
                            # Fallback to basic pandas reading
                            df = pd.read_csv(io.BytesIO(element.content))
                            delimiter = ','  # Default assumption
                        finally:
                            # Clean up temporary file
                            if 'tmp_file_path' in locals():
                                os.unlink(tmp_file_path)
                        
                    elif hasattr(element, 'path') and element.path:
                        # Use file path with CsvProcessor
                        try:
                            from csv_processor import CsvProcessor
                            processor = CsvProcessor()
                            result = processor.read_csv_with_options(element.path)
                            
                            if result['success']:
                                df = result['data']
                                delimiter = result['delimiter']
                            else:
                                await cl.Message(content=f"Error processing CSV: {result['error']}").send()
                                continue
                        except ImportError:
                            # Fallback to basic pandas reading
                            df = pd.read_csv(element.path)
                            delimiter = ','  # Default assumption
                    else:
                        await cl.Message(content="Could not access CSV file content.").send()
                        continue
                    
                    # Store dataframe in session for future use
                    cl.user_session.set("dataframe", df)
                    
                    # Display basic info about the CSV
                    info_message = f"""**CSV File Uploaded Successfully!**

📊 **Basic Information:**
- **File Name:** {element.name}
- **Rows:** {len(df)}
- **Columns:** {len(df.columns)}
- **Delimiter:** `{delimiter}`
- **Column Names:** {', '.join(df.columns.tolist())}"""
                    
                    await cl.Message(content=info_message).send()
                    
                    # Show preview
                    preview_message = "📋 **Data Preview (First 5 rows):**"
                    await cl.Message(content=preview_message).send()
                    
                    # Display first 5 rows as a table
                    await cl.Message(
                        content=f"```\n{df.head(5).to_string()}\n```"
                    ).send()
                    
                    # Start auto-detection process
                    await auto_detect_columns(df)
                    
                except Exception as e:
                    await cl.Message(content=f"Error processing CSV: {str(e)}").send()
            else:
                await cl.Message(
                    content="Please upload a CSV file. Other file types are not supported yet."
                ).send()
    else:
        # Handle regular chat messages
        df = cl.user_session.get("dataframe")
        column_mapping = cl.user_session.get("column_mapping")
        
        if df is None:
            await cl.Message(
                content="Please upload a CSV file first so I can help you analyze your data."
            ).send()
            return
        
        # If no column mapping, suggest proceeding to column selection
        if column_mapping is None:
            await cl.Message(
                content="Please complete the column selection process first to enable process mining analysis. You can also ask general questions about your data."
            ).send()
        
        user_message = message.content.lower()
        
        try:
            if "summary" in user_message or "statistics" in user_message:
                stats = df.describe()
                await cl.Message(
                    content=f"**Summary Statistics:**\n```\n{stats.to_string()}\n```"
                ).send()
            
            elif "process statistics" in user_message and column_mapping:
                case_col = column_mapping['case_id']
                activity_col = column_mapping['activity']
                
                total_cases = df[case_col].nunique()
                total_activities = len(df)
                unique_activities = df[activity_col].nunique()
                
                process_stats = f"""**Process Mining Statistics:**

📊 **Overview:**
- **Total Cases:** {total_cases}
- **Total Activity Instances:** {total_activities}
- **Unique Activities:** {unique_activities}
- **Average Activities per Case:** {total_activities / total_cases:.2f}

🔝 **Most Common Activities:**
{df[activity_col].value_counts().head().to_string()}
"""
                await cl.Message(content=process_stats).send()
            
            elif "columns" in user_message:
                await cl.Message(
                    content=f"**Columns in your dataset:**\n{', '.join(df.columns.tolist())}"
                ).send()
            
            elif "shape" in user_message or "size" in user_message:
                await cl.Message(
                    content=f"**Dataset Shape:** {df.shape[0]} rows × {df.shape[1]} columns"
                ).send()
            
            elif "preview" in user_message or "show data" in user_message:
                await cl.Message(
                    content=f"**Data Preview:**\n```\n{df.head(10).to_string()}\n```"
                ).send()
            
            else:
                help_message = "I can help you with:\n- Summary statistics ('show summary')\n- Column information ('show columns')\n- Dataset size ('show shape')\n- Data preview ('show preview')"
                
                if column_mapping:
                    help_message += "\n- Process statistics ('show process statistics')"
                
                help_message += "\n\nWhat would you like to know about your data?"
                
                await cl.Message(content=help_message).send()
                
        except Exception as e:
            await cl.Message(
                content=f"Sorry, I encountered an error: {str(e)}"
            ).send()