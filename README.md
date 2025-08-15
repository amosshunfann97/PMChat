# PMChat

A process mining chatbot with RAG (Retrieval-Augmented Generation) feature that allows users to upload CSV files and interact with process mining data through a conversational interface.

## Features

- **Process Mining Analysis**: Analyze process flows from CSV data
- **RAG-powered Chatbot**: Interactive conversations about your process data
- **CSV File Processing**: Upload and process manufacturing/business process data
- **Neo4j Integration**: Graph database for storing process relationships
- **Chainlit Interface**: User-friendly web interface for interactions

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amosshunfann97/PMChat.git
   cd PMChat
   ```

2. **Set up the environment:**
   ```bash
   cd pmchatbot
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python start_app.py
   ```

For detailed setup instructions, see the [pmchatbot README](./pmchatbot/README.md).

## Development Workflow

### Git Branch Management

When working on this project, you may need to manage Git branches. Here are the common commands for removing branches:

#### Removing Local Branches

1. **Delete a local branch (safe delete):**
   ```bash
   git branch -d branch-name
   ```
   This will only delete the branch if it has been fully merged.

2. **Force delete a local branch:**
   ```bash
   git branch -D branch-name
   ```
   Use this to delete unmerged branches (be careful!).

3. **Delete multiple local branches:**
   ```bash
   git branch -d branch1 branch2 branch3
   ```

#### Removing Remote Branches

1. **Delete a remote branch:**
   ```bash
   git push origin --delete branch-name
   ```

2. **Alternative syntax for deleting remote branch:**
   ```bash
   git push origin :branch-name
   ```

3. **Remove remote tracking references:**
   ```bash
   git remote prune origin
   ```
   This removes references to remote branches that no longer exist.

#### Common Workflows

1. **Clean up after feature completion:**
   ```bash
   # Switch to main branch
   git checkout main
   
   # Delete the local feature branch
   git branch -d feature-branch-name
   
   # Delete the remote feature branch
   git push origin --delete feature-branch-name
   ```

2. **Clean up stale tracking branches:**
   ```bash
   # See which remote branches are gone
   git remote show origin
   
   # Remove stale remote-tracking branches
   git remote prune origin
   ```

3. **List branches to see what needs cleanup:**
   ```bash
   # List all local branches
   git branch
   
   # List all remote branches
   git branch -r
   
   # List all branches (local and remote)
   git branch -a
   ```

### Before Deleting Branches

- Ensure you're not currently on the branch you want to delete
- Make sure important work is merged or backed up
- Consider using `git log --oneline branch-name` to review commits before deletion

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

When contributing:
1. Create a feature branch from `main`
2. Make your changes
3. Test thoroughly
4. Submit a pull request
5. Clean up your branches after merging (see Git Branch Management above)

## License

This project is licensed under the MIT License. 
