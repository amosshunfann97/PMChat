# Requirements Document

## Introduction

This feature integrates the existing process mining chatbot's core functionality (main.py) with the Chainlit frontend (app.py) to create a unified, interactive web-based process mining analysis tool. The integration will allow users to interact directly with the existing CSV data through part selection and perform comprehensive process mining analysis including process discovery, visualization, and GraphRAG-powered querying.

## Requirements

### Requirement 1

**User Story:** As a process analyst, I want to select specific parts from the existing process data through a web interface, so that I can focus my analysis on particular process variants without command-line interaction.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL load the existing CSV data and display available parts in a dropdown format with keyword search functionality
2. WHEN a user searches in the dropdown THEN the system SHALL filter the available parts based on the search keyword
3. WHEN a user selects a specific part from the dropdown THEN the system SHALL filter the data and process it using the existing PM4py pipeline from main.py
4. WHEN data processing is complete THEN the system SHALL store the processed chunks in Neo4j for retrieval
5. WHEN no part is selected THEN the system SHALL not allow process analysis to proceed and SHALL prompt the user to select a part

### Requirement 2

**User Story:** As a process analyst, I want to see both frequency DFG and performance DFG visualizations automatically displayed after part selection, so that I can immediately understand the process structure before asking questions.

#### Acceptance Criteria

1. WHEN data processing is complete for a selected part THEN the system SHALL automatically generate and display both frequency DFG and performance DFG visualizations
2. WHEN visualizations are displayed THEN the system SHALL show them in the chat interface before allowing user questions
3. WHEN DFG visualizations are displayed THEN the system SHALL provide zoom in and zoom out functionality for better readability
4. WHEN visualizations are generated THEN the system SHALL provide a brief summary of the process structure
5. WHEN the user is ready to ask questions THEN the system SHALL transition to the query interface

### Requirement 3

**User Story:** As a process analyst, I want to choose between OpenAI API and local LLM for generating responses, so that I can use the most appropriate model for my needs and environment.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL provide a toggle option to select between OpenAI API and local LLM
2. WHEN a user selects OpenAI API THEN the system SHALL display a popup window requesting the API key
3. WHEN an API key is provided THEN the system SHALL validate the key and configure the OpenAI client
4. WHEN a user selects local LLM THEN the system SHALL use the configured Ollama model
5. WHEN LLM selection is changed THEN the system SHALL update the configuration for subsequent queries

### Requirement 4

**User Story:** As a process analyst, I want to interact with my process data through natural language queries in the web interface, so that I can get insights without needing to know technical query syntax.

#### Acceptance Criteria

1. WHEN a user wants to ask a question THEN the system SHALL first prompt them to select a query context (activity, process, variant, or combined)
2. WHEN a query context is selected THEN the system SHALL allow the user to input their process-related question
3. WHEN a question is submitted THEN the system SHALL use the GraphRAG interface to retrieve relevant information from Neo4j without showing technical retrieval details
4. WHEN generating responses THEN the system SHALL use the selected LLM (OpenAI or Ollama) to provide natural language answers
5. WHEN a response is generated THEN the system SHALL require the user to select a new query context before asking their next question
6. WHEN responses are generated THEN the system SHALL display them in a conversational format without showing chunking results or retrieval scores

### Requirement 5

**User Story:** As a process analyst, I want to switch between different parts or return to part selection during my analysis session, so that I can compare different process variants within the same session.

#### Acceptance Criteria

1. WHEN analyzing a specific part THEN the system SHALL provide an option to return to part selection
2. WHEN switching between different parts THEN the system SHALL reprocess the data and update the Neo4j storage accordingly
3. WHEN returning to part selection THEN the system SHALL clear previous analysis context and require a new part selection
4. WHEN switching parts THEN the system SHALL maintain session continuity in the chat interface
5. WHEN a user types session termination keywords ('quit', 'exit', 'end', or 'stop') THEN the system SHALL end the current session and return to the initial state

### Requirement 6

**User Story:** As a system administrator, I want the integrated system to use the existing configuration and model settings, so that I can maintain consistent behavior across different interfaces.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL load configuration from the existing Config class without displaying technical model details to users
2. WHEN embeddings are required THEN the system SHALL use the configured embedding model transparently
3. WHEN reranking is enabled THEN the system SHALL apply the configured reranker model without showing technical details
4. IF configuration is invalid THEN the system SHALL provide clear error messages and fallback options
5. WHEN exporting data is requested THEN the system SHALL provide downloadable CSV files with process relationships