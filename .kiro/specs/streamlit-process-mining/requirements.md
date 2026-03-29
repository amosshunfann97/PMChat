# Requirements Document

## Introduction

This feature creates a modern Streamlit-based web interface for the existing process mining chatbot, replacing the Chainlit implementation with native Streamlit widgets for better user experience. The interface will provide intuitive dropdowns, toggles, and chat functionality while reusing the existing PM4py pipeline, Neo4j storage, and GraphRAG query system.

## Requirements

### Requirement 1

**User Story:** As a process analyst, I want to select parts using a native dropdown widget, so that I can easily browse and select from all available parts without complex button interfaces.

#### Acceptance Criteria

1. WHEN the application loads THEN the system SHALL display a native Streamlit selectbox containing all available parts from the CSV data
2. WHEN a user clicks the dropdown THEN the system SHALL show all parts in alphabetical order with search/filter capability
3. WHEN a user selects a part THEN the system SHALL immediately update the selection and trigger data processing
4. WHEN no part is selected THEN the system SHALL show a placeholder "Choose a part..." and disable analysis features
5. WHEN the part selection changes THEN the system SHALL clear previous analysis results and reprocess data

### Requirement 2

**User Story:** As a process analyst, I want to use native toggle and radio buttons for LLM and context selection, so that I can have a cleaner, more intuitive interface than action buttons.

#### Acceptance Criteria

1. WHEN the application loads THEN the system SHALL display a toggle switch for OpenAI API vs Local LLM selection
2. WHEN the OpenAI toggle is enabled THEN the system SHALL show a password input field for API key entry
3. WHEN the Local LLM option is selected THEN the system SHALL automatically configure Ollama without requiring additional input
4. WHEN a user selects query context THEN the system SHALL use radio buttons for Activity/Process/Variant/Combined selection
5. WHEN context selection changes THEN the system SHALL update the query routing accordingly

### Requirement 3

**User Story:** As a process analyst, I want a sidebar with all controls and a main area for chat and visualizations, so that I can have a clean, organized interface that doesn't clutter the analysis view.

#### Acceptance Criteria

1. WHEN the application loads THEN the system SHALL display all configuration controls in a collapsible sidebar
2. WHEN the sidebar is open THEN the system SHALL show part selection, LLM configuration, and query context controls
3. WHEN the main area loads THEN the system SHALL display the chat interface and visualizations without control clutter
4. WHEN visualizations are generated THEN the system SHALL display them in the main area with proper sizing and zoom capabilities
5. WHEN the sidebar is collapsed THEN the system SHALL maintain full functionality while maximizing analysis space

### Requirement 4

**User Story:** As a process analyst, I want a native Streamlit chat interface with message history, so that I can have a familiar chat experience with proper message threading and history.

#### Acceptance Criteria

1. WHEN a user types a message THEN the system SHALL use Streamlit's native chat input widget
2. WHEN messages are exchanged THEN the system SHALL display them using Streamlit's chat message containers with proper role indicators
3. WHEN the session continues THEN the system SHALL maintain chat history in session state
4. WHEN a response is generated THEN the system SHALL stream the response in real-time if possible
5. WHEN the session resets THEN the system SHALL clear chat history and start fresh

### Requirement 5

**User Story:** As a process analyst, I want automatic DFG visualizations displayed in organized tabs or columns, so that I can easily compare frequency and performance views side by side.

#### Acceptance Criteria

1. WHEN data processing completes THEN the system SHALL automatically generate both frequency and performance DFG visualizations
2. WHEN visualizations are ready THEN the system SHALL display them in organized tabs or columns for easy comparison
3. WHEN visualizations are displayed THEN the system SHALL provide zoom and download capabilities
4. WHEN new data is processed THEN the system SHALL update visualizations automatically
5. WHEN visualizations are large THEN the system SHALL provide proper scaling and scroll capabilities

### Requirement 6

**User Story:** As a process analyst, I want session state management that persists my selections and chat history, so that I can continue my analysis without losing progress during the session.

#### Acceptance Criteria

1. WHEN the application starts THEN the system SHALL initialize session state for all user selections and chat history
2. WHEN a user makes selections THEN the system SHALL persist them in session state across interactions
3. WHEN the user navigates or refreshes THEN the system SHALL maintain session state where possible
4. WHEN processing is in progress THEN the system SHALL show progress indicators and prevent conflicting actions
5. WHEN errors occur THEN the system SHALL maintain session state and provide recovery options

### Requirement 7

**User Story:** As a process analyst, I want the existing backend functionality (PM4py, Neo4j, GraphRAG) to work seamlessly with the new Streamlit interface, so that I can maintain all current analysis capabilities.

#### Acceptance Criteria

1. WHEN data processing is triggered THEN the system SHALL use the existing PM4py pipeline without modification
2. WHEN queries are submitted THEN the system SHALL use the existing GraphRAG retrievers and Neo4j storage
3. WHEN responses are generated THEN the system SHALL use the configured LLM (OpenAI or Ollama) as before
4. WHEN visualizations are created THEN the system SHALL use the existing visualization generation code
5. WHEN errors occur THEN the system SHALL use the existing error handling and logging systems

### Requirement 8

**User Story:** As a process analyst, I want enhanced UI features like progress bars, status indicators, and better error messages, so that I can have a more professional and informative user experience.

#### Acceptance Criteria

1. WHEN data processing starts THEN the system SHALL display progress bars and status messages
2. WHEN operations are in progress THEN the system SHALL show spinners and disable conflicting controls
3. WHEN errors occur THEN the system SHALL display user-friendly error messages with suggested actions
4. WHEN the system is ready for queries THEN the system SHALL show clear status indicators
5. WHEN operations complete THEN the system SHALL provide success notifications and next step guidance