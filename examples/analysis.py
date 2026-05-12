"""
Analyze results from ExperimentRunner CSV outputs.

Reads summary and detailed CSVs from out/ directory and produces
aggregated statistics for reporting.
"""

import pandas as pd
import glob
from pathlib import Path
import numpy as np


def analyze_summary_stats():
    """Aggregate summary CSVs to get mean/std of relative error per problem/solver."""
    summary_files = glob.glob("out/*.summary.out.csv")
    
    if not summary_files:
        print("No summary files found in out/")
        return None
    
    results = []
    
    for f in summary_files:
        df = pd.read_csv(f)
        
        # Extract problem and solver from filename since columns may vary
        filename = Path(f).stem  # removes .summary.out and .csv
        # Expected: {problem}_{solver}.summary.out
        parts = filename.rsplit('_', 1)
        if len(parts) == 2:
            problem = parts[0]
            solver = parts[1]
        else:
            print(f"Warning: Could not parse filename {f}")
            continue
        
        # Try to get relative_error_distance, fall back to other column names
        if 'relative_error_distance' in df.columns:
            red_values = df['relative_error_distance'].dropna()
        elif 'P_RED' in df.columns:
            red_values = df['P_RED'].dropna()
        else:
            print(f"Warning: Could not find relative error column in {f}. Columns: {list(df.columns)}")
            continue
        
        if len(red_values) > 0:
            mean_red = red_values.mean()
            std_red = red_values.std()
            num_runs = len(red_values)
            
            results.append({
                'Problem': problem,
                'Solver': solver,
                'Num_Runs': num_runs,
                'Mean_P_RED': mean_red,
                'Std_P_RED': std_red,
                'Min_P_RED': red_values.min(),
                'Max_P_RED': red_values.max(),
            })
    
    if results:
        results_df = pd.DataFrame(results)
        return results_df
    return None


def analyze_feasibility():
    """Extract final feasibility % per problem/solver/run from detailed CSVs."""
    detail_files = glob.glob("out/*.out.csv")
    
    if not detail_files:
        print("No detail files found in out/")
        return None
    
    results = []
    
    for f in detail_files:
        # Skip summary files
        if 'summary' in f:
            continue
        
        df = pd.read_csv(f)
        
        # Extract problem and solver name from filename
        # Expected format: out/{problem}_{solver}.out.csv
        filename = Path(f).stem  # removes .csv and path
        parts = filename.rsplit('_', 1)  # split from right on last underscore
        if len(parts) == 2:
            problem = parts[0]
            solver = parts[1]
        else:
            print(f"Warning: Could not parse filename {f}")
            continue
        
        # Check what columns are available
        if 'feasibility' not in df.columns:
            print(f"Warning: 'feasibility' column not found in {f}. Columns: {list(df.columns)}")
            continue
        
        # Get last iteration per run (final state)
        if 'run_id' in df.columns:
            final_iters = df.groupby('run_id').tail(1)
        else:
            # If no run_id, treat entire file as one run
            final_iters = df.tail(1)
            final_iters['run_id'] = 1
        
        for _, row in final_iters.iterrows():
            results.append({
                'Problem': problem,
                'Solver': solver,
                'Run_ID': int(row['run_id']) if 'run_id' in row else 1,
                'Final_Feasibility_Pct': row['feasibility'],
                'Final_Fitness': row['accuracy'] if 'accuracy' in row.index else np.nan,
            })
    
    if results:
        results_df = pd.DataFrame(results)
        return results_df
    return None


def analyze_feasibility_stats():
    """Aggregate feasibility statistics by problem/solver."""
    feas_df = analyze_feasibility()
    
    if feas_df is None:
        return None
    
    # Group by problem and solver
    grouped = feas_df.groupby(['Problem', 'Solver'])['Final_Feasibility_Pct'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    grouped.columns = ['Problem', 'Solver', 'Mean_Feasibility_Pct', 'Std_Feasibility_Pct', 
                       'Min_Feasibility_Pct', 'Max_Feasibility_Pct', 'Num_Runs']
    
    return grouped



def analyze_fitness_stats():
    """Aggregate final fitness statistics by problem/solver."""
    feas_df = analyze_feasibility()

    if feas_df is None:
        return None

    grouped = feas_df.groupby(['Problem', 'Solver'])['Final_Fitness'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()

    grouped.columns = ['Problem', 'Solver', 'Mean_Final_Fitness', 'Std_Final_Fitness',
                       'Min_Final_Fitness', 'Max_Final_Fitness', 'Num_Runs']

    return grouped

def analyze_convergence_per_problem():
    """Track convergence over iterations (diversity, feasibility trend)."""
    detail_files = glob.glob("out/*.out.csv")
    
    if not detail_files:
        return None
    
    results = []
    
    for f in detail_files:
        if 'summary' in f:
            continue
        
        df = pd.read_csv(f)
        
        # Extract from filename
        filename = Path(f).stem
        parts = filename.rsplit('_', 1)
        if len(parts) == 2:
            problem = parts[0]
            solver = parts[1]
        else:
            continue
        
        # Check for required columns
        if 'iteration' not in df.columns or 'accuracy' not in df.columns:
            print(f"Warning: Missing required columns in {f}")
            continue
        
        run_col = 'run_id' if 'run_id' in df.columns else None
        
        run_ids = df[run_col].unique() if run_col else [1]
        
        for run_id in run_ids:
            if run_col:
                run_data = df[df[run_col] == run_id].sort_values('iteration')
            else:
                run_data = df.sort_values('iteration')
            
            # Sample every 100 iterations to reduce noise
            run_data_sampled = run_data[run_data['iteration'] % 100 == 0].copy()
            
            if len(run_data_sampled) > 0:
                # Get final iteration stats
                final = run_data_sampled.iloc[-1]
                
                # Get improvement from iteration 100 to final
                early = run_data_sampled[run_data_sampled['iteration'] <= 100]
                if len(early) > 0:
                    early_accuracy = early.iloc[-1]['accuracy']
                    final_accuracy = final['accuracy']
                    improvement = final_accuracy - early_accuracy
                else:
                    early_accuracy = np.nan
                    final_accuracy = final['accuracy']
                    improvement = np.nan
                
                results.append({
                    'Problem': problem,
                    'Solver': solver,
                    'Run_ID': run_id,
                    'Final_Iteration': int(final['iteration']),
                    'Final_Diversity': final['diversity'] if 'diversity' in final.index else np.nan,
                    'Final_Feasibility': final['feasibility'] if 'feasibility' in final.index else np.nan,
                    'Early_Accuracy': early_accuracy,
                    'Final_Accuracy': final_accuracy,
                    'Improvement': improvement,
                })
    
    if results:
        return pd.DataFrame(results)
    return None


def print_report():
    """Generate a formatted report of all analyses."""
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("="*100)
    
    # 1. Summary of relative error
    print("\n1. RELATIVE ERROR DISTANCE (P_RED) - Lower is better")
    print("-" * 100)
    summary_stats = analyze_summary_stats()
    if summary_stats is not None:
        # Sort by problem name for readability
        summary_stats = summary_stats.sort_values('Problem')
        print(summary_stats.to_string(index=False))
        print(f"\nOverall mean P_RED: {summary_stats['Mean_P_RED'].mean():.4f}")
    else:
        print("No summary statistics available.")
    
    # 2. Feasibility statistics
    print("\n\n2. FEASIBILITY - % of population meeting constraints (Higher is better)")
    print("-" * 100)
    feas_stats = analyze_feasibility_stats()
    if feas_stats is not None:
        feas_stats = feas_stats.sort_values('Problem')
        # Format for readability
        feas_display = feas_stats.copy()
        feas_display['Mean_Feasibility_Pct'] = feas_display['Mean_Feasibility_Pct'].round(2)
        feas_display['Std_Feasibility_Pct'] = feas_display['Std_Feasibility_Pct'].round(2)
        feas_display['Min_Feasibility_Pct'] = feas_display['Min_Feasibility_Pct'].round(2)
        feas_display['Max_Feasibility_Pct'] = feas_display['Max_Feasibility_Pct'].round(2)
        print(feas_display.to_string(index=False))
    else:
        print("No feasibility data available.")
    
    # 2b. Fitness statistics
    print("\n\n2b. FINAL FITNESS STATISTICS")
    print("-" * 100)
    fitness_stats = analyze_fitness_stats()
    if fitness_stats is not None:
        fitness_stats = fitness_stats.sort_values('Problem')
        fitness_display = fitness_stats.copy()
        for col in ['Mean_Final_Fitness', 'Std_Final_Fitness', 'Min_Final_Fitness', 'Max_Final_Fitness']:
            fitness_display[col] = fitness_display[col].round(4)
        print(fitness_display.to_string(index=False))
    else:
        print("No fitness data available.")

    # 3. Convergence analysis
    print("\n\n3. CONVERGENCE ANALYSIS - Improvement from early to final iterations")
    print("-" * 100)
    fitness_stats = analyze_fitness_stats()
    if fitness_stats is not None:
        fitness_stats.to_csv("analysis_fitness_summary.csv", index=False)
        print("Saved fitness summary to: analysis_fitness_summary.csv")

    conv_df = analyze_convergence_per_problem()
    if conv_df is not None:
        conv_summary = conv_df.groupby(['Problem', 'Solver']).agg({
            'Final_Diversity': 'mean',
            'Final_Feasibility': 'mean',
            'Improvement': 'mean',
        }).reset_index()
        conv_summary = conv_summary.sort_values('Problem')
        conv_summary.columns = ['Problem', 'Solver', 'Mean_Final_Diversity', 
                                'Mean_Final_Feasibility', 'Mean_Improvement']
        for col in ['Mean_Final_Diversity', 'Mean_Final_Feasibility', 'Mean_Improvement']:
            conv_summary[col] = conv_summary[col].round(4)
        print(conv_summary.to_string(index=False))
    else:
        print("No convergence data available.")
    
    print("\n" + "="*100)


def export_tables_for_report():
    """Export tables as formatted strings for easy copy-paste into report."""
    print("\n" + "="*100)
    print("FORMATTED TABLES FOR REPORT")
    print("="*100)
    
    # Table 1: P_RED summary
    print("\nTable 1: Relative Error Distance (P_RED) Summary")
    print("-" * 100)
    summary_stats = analyze_summary_stats()
    if summary_stats is not None:
        summary_stats = summary_stats.sort_values('Problem')
        summary_stats['Mean±Std'] = (
            summary_stats['Mean_P_RED'].round(4).astype(str) + 
            ' ± ' + 
            summary_stats['Std_P_RED'].round(4).astype(str)
        )
        table1 = summary_stats[['Problem', 'Solver', 'Num_Runs', 'Mean±Std', 'Min_P_RED', 'Max_P_RED']]
        table1.columns = ['Problem', 'Solver', 'Runs', 'Mean P_RED ± Std', 'Min', 'Max']
        print(table1.to_string(index=False))
    
    # Table 2: Feasibility summary
    print("\n\nTable 2: Final Feasibility Summary")
    print("-" * 100)
    feas_stats = analyze_feasibility_stats()
    if feas_stats is not None:
        feas_stats = feas_stats.sort_values('Problem')
        feas_stats['Mean±Std'] = (
            feas_stats['Mean_Feasibility_Pct'].round(2).astype(str) + 
            '% ± ' + 
            feas_stats['Std_Feasibility_Pct'].round(2).astype(str) + '%'
        )
        table2 = feas_stats[['Problem', 'Solver', 'Num_Runs', 'Mean±Std', 'Min_Feasibility_Pct', 'Max_Feasibility_Pct']]
        table2.columns = ['Problem', 'Solver', 'Runs', 'Mean Feasibility ± Std', 'Min %', 'Max %']
        print(table2.to_string(index=False))
    
    # Table 3: Fitness summary
    print("\n\nTable 3: Final Fitness Summary")
    print("-" * 100)
    fitness_stats = analyze_fitness_stats()
    if fitness_stats is not None:
        fitness_stats = fitness_stats.sort_values('Problem')
        fitness_stats['Mean±Std'] = (
            fitness_stats['Mean_Final_Fitness'].round(4).astype(str) +
            ' ± ' +
            fitness_stats['Std_Final_Fitness'].round(4).astype(str)
        )
        table3 = fitness_stats[['Problem', 'Solver', 'Num_Runs', 'Mean±Std', 'Min_Final_Fitness', 'Max_Final_Fitness']]
        table3.columns = ['Problem', 'Solver', 'Runs', 'Mean Fitness ± Std', 'Min', 'Max']
        print(table3.to_string(index=False))

    print("\n" + "="*100)


if __name__ == "__main__":
    # Make sure out/ directory exists
    Path("out").mkdir(exist_ok=True)
    
    # Generate full report
    print_report()
    
    # Generate formatted tables
    export_tables_for_report()
    
    # Also save to CSV for reference
    summary_stats = analyze_summary_stats()
    if summary_stats is not None:
        summary_stats.to_csv("analysis_p_red_summary.csv", index=False)
        print("\nSaved P_RED summary to: analysis_p_red_summary.csv")
    
    feas_stats = analyze_feasibility_stats()
    if feas_stats is not None:
        feas_stats.to_csv("analysis_feasibility_summary.csv", index=False)
        print("Saved feasibility summary to: analysis_feasibility_summary.csv")
    
    fitness_stats = analyze_fitness_stats()
    if fitness_stats is not None:
        fitness_stats.to_csv("analysis_fitness_summary.csv", index=False)
        print("Saved fitness summary to: analysis_fitness_summary.csv")

    conv_df = analyze_convergence_per_problem()
    if conv_df is not None:
        conv_df.to_csv("analysis_convergence_detail.csv", index=False)
        print("Saved convergence detail to: analysis_convergence_detail.csv")