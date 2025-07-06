import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
from typing import Dict, Tuple, List, Optional

class DirectlyFollowsGraphGenerator:
    """
    A class to generate and visualize directly-follows graphs from process mining data
    using pm4py and your local CSV files.
    """
    
    def __init__(self, csv_file_path: str):
        """
        Initialize the DFG generator with a CSV file path.
        
        Args:
            csv_file_path (str): Path to the CSV file containing process mining data
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.log = None
        self.dfg = None
        self.start_activities = None
        self.end_activities = None
        
    def load_data(self, case_col: str = None, activity_col: str = None, 
                  timestamp_col: str = None, separator: str = ';') -> pd.DataFrame:
        """
        Load and prepare data from CSV file.
        
        Args:
            case_col (str): Name of the case ID column
            activity_col (str): Name of the activity column  
            timestamp_col (str): Name of the timestamp column
            separator (str): CSV separator (default: ';')
            
        Returns:
            pd.DataFrame: Loaded and prepared dataframe
        """
        print(f"Loading data from {self.csv_file_path}...")
        
        try:
            # Load CSV file
            self.df = pd.read_csv(self.csv_file_path, sep=separator)
            print(f"✅ Loaded dataset with {len(self.df)} events")
            
            # Auto-detect columns if not specified
            if not case_col or not activity_col or not timestamp_col:
                case_col, activity_col, timestamp_col = self._auto_detect_columns()
            
            # Standardize column names
            column_mapping = {}
            if case_col and case_col in self.df.columns:
                column_mapping[case_col] = 'case_id'
            if activity_col and activity_col in self.df.columns:
                column_mapping[activity_col] = 'activity'
            if timestamp_col and timestamp_col in self.df.columns:
                column_mapping[timestamp_col] = 'timestamp'
                
            self.df = self.df.rename(columns=column_mapping)
            
            # Convert data types
            if 'case_id' in self.df.columns:
                self.df['case_id'] = self.df['case_id'].astype(str)
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                
            # Sort by case and timestamp
            self.df = self.df.sort_values(['case_id', 'timestamp'])
            
            print(f"📊 Dataset info:")
            print(f"   - Cases: {self.df['case_id'].nunique()}")
            print(f"   - Activities: {self.df['activity'].nunique()}")
            print(f"   - Events: {len(self.df)}")
            print(f"   - Activity types: {list(self.df['activity'].unique())}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def _auto_detect_columns(self) -> Tuple[str, str, str]:
        """Auto-detect case, activity, and timestamp columns."""
        columns = self.df.columns.tolist()
        
        # Common patterns for each column type
        case_patterns = ['case', 'case_id', 'caseid', 'case:concept:name']
        activity_patterns = ['activity', 'concept:name', 'event', 'task']
        timestamp_patterns = ['timestamp', 'time', 'datetime', 'time:timestamp']
        
        case_col = self._find_column_by_patterns(columns, case_patterns)
        activity_col = self._find_column_by_patterns(columns, activity_patterns)
        timestamp_col = self._find_column_by_patterns(columns, timestamp_patterns)
        
        print(f"🔍 Auto-detected columns:")
        print(f"   - Case ID: {case_col}")
        print(f"   - Activity: {activity_col}")
        print(f"   - Timestamp: {timestamp_col}")
        
        return case_col, activity_col, timestamp_col
    
    def _find_column_by_patterns(self, columns: List[str], patterns: List[str]) -> Optional[str]:
        """Find column name that matches given patterns."""
        for pattern in patterns:
            for col in columns:
                if pattern.lower() in col.lower():
                    return col
        return columns[0] if columns else None
    
    def prepare_pm4py_log(self) -> pm4py.objects.log.obj.EventLog:
        """Convert DataFrame to PM4py event log format."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("🔄 Converting to PM4py event log...")
        
        # Create a copy for PM4py format
        log_df = self.df.copy()
        
        # Rename columns to PM4py standard format
        log_df = log_df.rename(columns={
            'case_id': 'case:concept:name',
            'activity': 'concept:name',
            'timestamp': 'time:timestamp'
        })
        
        # Convert to PM4py event log
        self.log = pm4py.convert_to_event_log(log_df)
        print(f"✅ Event log created with {len(self.log)} traces")
        
        return self.log
    
    def discover_dfg(self) -> Tuple[Dict, Dict, Dict]:
        """
        Discover directly-follows graph using PM4py.
        
        Returns:
            Tuple containing DFG, start activities, and end activities
        """
        if self.log is None:
            raise ValueError("Event log not prepared. Call prepare_pm4py_log() first.")
        
        print("🔍 Discovering directly-follows graph...")
        
        # Discover DFG
        self.dfg, self.start_activities, self.end_activities = pm4py.discover_dfg(self.log)
        
        print(f"📈 DFG Discovery Results:")
        print(f"   - DFG edges: {len(self.dfg)}")
        print(f"   - Start activities: {len(self.start_activities)}")
        print(f"   - End activities: {len(self.end_activities)}")
        
        # Print detailed results
        print(f"\n🎯 Start activities:")
        for activity, count in self.start_activities.items():
            print(f"   - {activity}: {count} times")
            
        print(f"\n🏁 End activities:")
        for activity, count in self.end_activities.items():
            print(f"   - {activity}: {count} times")
            
        print(f"\n➡️ Directly-follows relationships:")
        for (source, target), count in sorted(self.dfg.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {source} → {target}: {count} times")
        
        return self.dfg, self.start_activities, self.end_activities
    
    def visualize_dfg_pm4py(self, output_path: str = None) -> None:
        """
        Visualize DFG using PM4py's built-in visualization.
        
        Args:
            output_path (str): Path to save the visualization (optional)
        """
        if self.dfg is None:
            raise ValueError("DFG not discovered. Call discover_dfg() first.")
        
        print("🎨 Creating PM4py DFG visualization...")
        
        try:
            # Use PM4py's DFG visualization
            pm4py.view_dfg(self.dfg, self.start_activities, self.end_activities)
            
            if output_path:
                pm4py.save_vis_dfg(self.dfg, self.start_activities, self.end_activities, output_path)
                print(f"💾 DFG saved to {output_path}")
                
        except Exception as e:
            print(f"⚠️ PM4py visualization failed: {e}")
            print("Falling back to custom visualization...")
            self.visualize_dfg_custom(output_path)
    
    def visualize_dfg_custom(self, output_path: str = None, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create custom DFG visualization using NetworkX and Matplotlib.
        
        Args:
            output_path (str): Path to save the visualization
            figsize (tuple): Figure size (width, height)
        """
        if self.dfg is None:
            raise ValueError("DFG not discovered. Call discover_dfg() first.")
        
        print("🎨 Creating custom DFG visualization...")
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes (activities)
        all_activities = set()
        for (source, target), count in self.dfg.items():
            all_activities.add(source)
            all_activities.add(target)
        
        for activity in all_activities:
            G.add_node(activity)
        
        # Add edges (directly-follows relationships)
        for (source, target), count in self.dfg.items():
            G.add_edge(source, target, weight=count)
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        # Calculate layout
        pos = self._calculate_layout(G)
        
        # Draw nodes
        self._draw_nodes(G, pos)
        
        # Draw edges
        self._draw_edges(G, pos)
        
        # Add labels
        self._add_labels(G, pos)
        
        plt.title("Directly-Follows Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"💾 Custom DFG saved to {output_path}")
        
        plt.show()
    
    def _calculate_layout(self, G: nx.DiGraph) -> Dict:
        """Calculate optimal layout for the graph."""
        try:
            # Try hierarchical layout first
            pos = nx.spring_layout(G, k=3, iterations=50)
        except:
            # Fallback to circular layout
            pos = nx.circular_layout(G)
        
        return pos
    
    def _draw_nodes(self, G: nx.DiGraph, pos: Dict) -> None:
        """Draw nodes with different colors for start/end activities."""
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            if node in self.start_activities:
                node_colors.append('lightgreen')
                node_sizes.append(3000)
            elif node in self.end_activities:
                node_colors.append('lightcoral')
                node_sizes.append(3000)
            else:
                node_colors.append('lightblue')
                node_sizes.append(2000)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                              alpha=0.8, linewidths=2, edgecolors='black')
    
    def _draw_edges(self, G: nx.DiGraph, pos: Dict) -> None:
        """Draw edges with thickness proportional to frequency."""
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Normalize weights for edge thickness
        if weights:
            max_weight = max(weights)
            edge_widths = [3 * (w / max_weight) + 0.5 for w in weights]
        else:
            edge_widths = [1] * len(edges)
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7,
                              edge_color='gray', arrowsize=20, arrowstyle='->')
        
        # Add edge labels (frequency)
        edge_labels = {(u, v): str(G[u][v]['weight']) for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    def _add_labels(self, G: nx.DiGraph, pos: Dict) -> None:
        """Add activity labels to nodes."""
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    def export_dfg_data(self, output_path: str = "dfg_data.csv") -> None:
        """
        Export DFG data to CSV file.
        
        Args:
            output_path (str): Path to save the CSV file
        """
        if self.dfg is None:
            raise ValueError("DFG not discovered. Call discover_dfg() first.")
        
        print(f"💾 Exporting DFG data to {output_path}...")
        
        # Prepare data for export
        dfg_data = []
        
        for (source, target), count in self.dfg.items():
            dfg_data.append({
                'source_activity': source,
                'target_activity': target,
                'frequency': count,
                'relationship_type': 'directly_follows'
            })
        
        # Add start activities
        for activity, count in self.start_activities.items():
            dfg_data.append({
                'source_activity': 'START',
                'target_activity': activity,
                'frequency': count,
                'relationship_type': 'start_activity'
            })
        
        # Add end activities
        for activity, count in self.end_activities.items():
            dfg_data.append({
                'source_activity': activity,
                'target_activity': 'END',
                'frequency': count,
                'relationship_type': 'end_activity'
            })
        
        # Save to CSV
        dfg_df = pd.DataFrame(dfg_data)
        dfg_df.to_csv(output_path, index=False)
        print(f"✅ DFG data exported with {len(dfg_df)} relationships")
    
    def print_dfg_statistics(self) -> None:
        """Print detailed statistics about the discovered DFG."""
        if self.dfg is None:
            raise ValueError("DFG not discovered. Call discover_dfg() first.")
        
        print("\n📊 DFG Statistics:")
        print("=" * 50)
        
        # Basic stats
        total_edges = len(self.dfg)
        total_frequency = sum(self.dfg.values())
        
        print(f"Total directly-follows relationships: {total_edges}")
        print(f"Total frequency: {total_frequency}")
        
        if total_edges > 0:
            avg_frequency = total_frequency / total_edges
            print(f"Average frequency per relationship: {avg_frequency:.2f}")
        
        # Most frequent relationships
        print(f"\n🔝 Top 5 most frequent relationships:")
        sorted_dfg = sorted(self.dfg.items(), key=lambda x: x[1], reverse=True)
        for i, ((source, target), count) in enumerate(sorted_dfg[:5], 1):
            percentage = (count / total_frequency) * 100
            print(f"   {i}. {source} → {target}: {count} ({percentage:.1f}%)")
        
        # Activity statistics
        activities = set()
        for (source, target), _ in self.dfg.items():
            activities.add(source)
            activities.add(target)
        
        print(f"\n🎯 Activity statistics:")
        print(f"   - Total unique activities: {len(activities)}")
        print(f"   - Start activities: {len(self.start_activities)}")
        print(f"   - End activities: {len(self.end_activities)}")


def main():
    """Example usage of the DirectlyFollowsGraphGenerator."""
    print("🚀 Starting Directly-Follows Graph Generation")
    print("=" * 60)
    
    # File path - adjust this to your CSV file
    csv_file_path = r"C:\Users\shunf\RoadToMaster\PMChat\running_example_manufacturing.csv"
    
    try:
        # Initialize generator
        dfg_gen = DirectlyFollowsGraphGenerator(csv_file_path)
        
        # Load data (auto-detect columns)
        dfg_gen.load_data()
        
        # Prepare PM4py log
        dfg_gen.prepare_pm4py_log()
        
        # Discover DFG
        dfg_gen.discover_dfg()
        
        # Print statistics
        dfg_gen.print_dfg_statistics()
        
        # Create visualizations
        print(f"\n🎨 Creating visualizations...")
        
        # Try PM4py visualization only
        try:
            dfg_gen.visualize_dfg_pm4py("dfg_pm4py.png")
        except:
            print("PM4py visualization not available.")
        
        # Export data
        dfg_gen.export_dfg_data("dfg_relationships.csv")
        
        print(f"\n✅ DFG generation completed successfully!")
        print(f"📁 Generated files:")
        print(f"   - dfg_pm4py.png (PM4py visualization, if available)")
        print(f"   - dfg_relationships.csv (data export)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()