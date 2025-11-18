import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import pandas as pd
from scipy.stats import ttest_ind

# set random seed for reproducibility
np.random.seed(20251212)

def simulate_rw_mc(
    task_matrix, 
    N,
    partial_full, # 1 = full feedback, 0 = partial
    outcome_mode, # 'standard' or 'history'
):
    n_trials, n_opts = task_matrix.shape
    convergence_trials = []
    all_choices = np.zeros((N, n_trials), dtype=int)
    all_switches = np.zeros((N, n_trials, n_opts))
    switch_counts = np.zeros(N, dtype=int)
    selected_counts = np.zeros((n_trials, n_opts))
    convergent_direction = np.full(N, -1) # -1 for no convergence

    for sim in range(N):
        # Sample alpha, gamma, theta uniformly for each simulation
        alpha = np.random.uniform(0.05, 0.15) 
        gamma = np.random.uniform(0.3, 0.7)
        theta = np.random.uniform(0.3, 0.7)
        history_weight = np.random.uniform(0.3, 0.7)
        theta_2e = np.random.uniform(-1, 1)
        E = np.zeros((n_trials + 1, n_opts))
        visible_outcome = [[] for _ in range(n_opts)]
        choices = np.zeros(n_trials, dtype=int)
        delta_mat = partial_full * np.ones((n_trials, n_opts))
        for t in range(n_trials):
            # theta = 2**theta_2e * (t + 1) / 8
            p = np.exp(theta * E[t]) / np.sum(np.exp(theta * E[t]))
            if t == 0:
                choice = 0
            else:
                choice = np.random.choice([0, 1], p=p)
            choices[t] = choice
            for j in [0, 1]:
                if partial_full == 1:
                    visible_outcome[j].append(task_matrix[t][j])
                else:
                    if j == choice:
                        visible_outcome[j].append(task_matrix[t][j])
            for j in [0, 1]:
                if outcome_mode == 'standard':
                    outcome = task_matrix[t][j]
                elif outcome_mode == 'history':
                    prev_vis = visible_outcome[j][:-1] if t > 0 else []
                    history_avg = np.mean(prev_vis) if prev_vis else 0
                    outcome = history_weight * history_avg + (1 - history_weight) * task_matrix[t][j]
                else:
                    raise ValueError('outcome_mode must be "standard" or "history"')
                delta = delta_mat[t][j]
                E[t + 1][j] = E[t][j] + alpha * (delta + gamma * (1 - delta)) * (outcome - E[t][j])
        all_choices[sim] = choices
        for j in [0, 1]:
            switched = np.zeros(n_trials)
            switched[1:] = (choices[1:] != choices[:-1]) & (choices[1:] == j)
            all_switches[sim, :, j] = switched
            selected_counts[:, j] += (choices == j)
        convergence_found = False
        converged_to = -1
        for t in range(2, n_trials - 1):
            subseq = choices[t:]
            maj = np.bincount(subseq).argmax()
            exceptions = np.sum(subseq != maj)
            if exceptions <= 1 and not (t >= 13):
                convergence_trials.append(t)
                convergent_direction[sim] = maj  # 0 or 1: direction to which it converged
                convergence_found = True
                converged_to = maj
                break
        # for a simulation, count the switch count across all trials
        switch_counts[sim] = np.sum(choices[1:] != choices[:-1])
        if not convergence_found or (convergence_found and t >= 13):
            convergence_trials.append(13) # Mark 14 as 'no convergence' if none found
            convergent_direction[sim] = -1 # No convergence
    
    percent_selected = 100.0 * selected_counts / N
    mean_switch = np.mean(all_switches, axis=0)
    valid_ct = [c for c in convergence_trials if c != 13]
    no_conv_ct = [c for c in convergence_trials if c == 13]
    if valid_ct:
        mean_conv = np.mean(valid_ct)
        se = np.std(valid_ct, ddof=1) / np.sqrt(len(valid_ct))
        ci_low, ci_high = norm.interval(0.95, loc=mean_conv, scale=se)
    else:
        mean_conv, ci_low, ci_high = np.nan, np.nan, np.nan

    # Convergent direction descriptive statistics
    n_converged = np.sum(convergent_direction != -1)
    n_conv_0 = np.sum(convergent_direction == 0)
    n_conv_1 = np.sum(convergent_direction == 1)
    percent_conv_0 = 100.0 * n_conv_0 / n_converged if n_converged > 0 else 0
    percent_conv_1 = 100.0 * n_conv_1 / n_converged if n_converged > 0 else 0

    # print the results summary, including the overall percent selected for each option (from 2nd to 16th, 1-indexed), 
    # the convergence time and its discriptives, the mean switch rate for each option,
    # and the descriptive convergence to each direction
    print('--- Simulation Results Summary ---')
    print(f'Overall Percent Selected (Option 1/2): {np.mean(percent_selected[1:], axis=0)}')
    print(f"% Converged to Option 1: {percent_conv_0:.1f}%")
    print(f"% Converged to Option 2: {percent_conv_1:.1f}%")
    print(f"% No convergence: {100.0 * len(no_conv_ct) / N:.1f}%")
    print(f'Mean Convergence Trial: {mean_conv:.2f}')
    print(f'95% CI for Convergence Trial: [{ci_low:.2f}, {ci_high:.2f}]')
    print(f"Mean Switch Rate (Option 1/2): {mean_switch[-1]}")
    print('----------------------------------')

    return {
        'convergence_trials': convergence_trials,
        'mean_convergence': mean_conv,
        'ci': (ci_low, ci_high),
        'all_choices': all_choices,
        'all_switches': all_switches,
        "switch_counts": switch_counts,
        'percent_selected': percent_selected,
        'mean_switch': mean_switch,
        'no_conv_ct': len(no_conv_ct),
        'convergent_direction': convergent_direction
    }

def plot_rw_results(results, n_trials, task_name, file_suffix):
    percent_selected = results['percent_selected']
    mean_switch = results['mean_switch']
    convergence_trials = results['convergence_trials']
    mean_conv = results['mean_convergence']
    ci_low, ci_high = results['ci']
    no_conv_ct = results['no_conv_ct']

    plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1])
    # Earliest Convergence Trial
    ax_left = plt.subplot(gs[:, 0])
    # 1. Make t 1-indexed: add 1 to all t values
    valid_ct = [c + 1 for c in convergence_trials if c < 13]
    # 2. Restrict X-axis to 3~15 (inclusive)
    trial_vals = list(range(3, 16))
    counts = [valid_ct.count(t) for t in trial_vals]
    ax_left.bar(trial_vals, counts, color='limegreen', alpha=0.7)
    # Adjust mean/CI for 1-indexing
    if mean_conv == mean_conv:  # Check for nan
        ax_left.axvline(mean_conv + 1, color='k', ls='--', label=f'Mean: {mean_conv+1:.1f}')
        ax_left.axvspan(ci_low + 1, ci_high + 1, color='orange', alpha=0.3, label=f'95% CI: [{ci_low+1:.1f},{ci_high+1:.1f}]')
    ax_left.axvline(13.5, color='red', ls='--', label='No convergence')
    ax_left.text(13.5, ax_left.get_ylim()[1]/2, f'No convergence\np={no_conv_ct/len(convergence_trials) * 100:.2f}%', color='red', ha='center', va='center')
    ax_left.set_xlabel('Convergence trial')
    ax_left.set_ylabel('Frequency')
    ax_left.set_title(f'{task_name}: Earliest Convergence Trial')
    ax_left.set_xlim(3, 14)
    ax_left.legend()
    # Switch frequency for option 1
    ax_mid_up = plt.subplot(gs[0, 1])
    ax_mid_up.plot(range(1, n_trials + 1), mean_switch[:, 0], marker='o', label='Switch freq: Option 1')
    ax_mid_up.set_title('Switch Frequency: Option 1')
    ax_mid_up.set_xlabel('Trial')
    ax_mid_up.set_ylabel('Switch Rate')
    ax_mid_up.set_ylim(0, 1)
    ax_mid_up.grid(True)
    ax_mid_up.legend()
    # Switch frequency for option 2
    ax_mid_down = plt.subplot(gs[1, 1])
    ax_mid_down.plot(range(1, n_trials + 1), mean_switch[:, 1], marker='o', color='r', label='Switch freq: Option 2')
    ax_mid_down.set_title('Switch Frequency: Option 2')
    ax_mid_down.set_xlabel('Trial')
    ax_mid_down.set_ylabel('Switch Rate')
    ax_mid_down.set_ylim(0, 1)
    ax_mid_down.grid(True)
    ax_mid_down.legend()
    # Percent chosen
    ax_right = plt.subplot(gs[:, 2])
    ax_right.fill_between(range(1, n_trials + 1), percent_selected[:, 0], color='blue', alpha=0.6, label='Option 1')
    ax_right.fill_between(range(1, n_trials + 1), percent_selected[:, 0], 100, color='orange', alpha=0.5, label='Option 2')
    ax_right.plot(range(1, n_trials + 1), percent_selected[:, 0], color='navy')
    ax_right.set_ylim(0, 100)
    ax_right.set_title(f'{task_name}: Choice Distribution (%)')
    ax_right.set_xlabel('Trial')
    ax_right.set_ylabel('% of Simulations (Option 1/2)')
    ax_right.legend()
    plt.tight_layout()
    plt.savefig(f'{task_name.replace(" ", "_").lower()}_{file_suffix}_results.png')
    plt.close()

tasks = []
# Game 2 early 20 blocks from Rakow & Miler (2009), left is NS, right is S
task1 = np.array([[10 if np.random.rand() < 0.9 else -20, 10 if np.random.rand() < 0.7 else -20] for _ in range(16)])
# Game 3 early 20 blocks from Rakow & Miler (2009)
task2 = np.array([[20 if np.random.rand() < 0.1 else -10, 20 if np.random.rand() < 0.3 else -10] for _ in range(16)])
# Left always 3, right has 10% chance of 32, 90% chance of 0 (from Hertwig et al., 2004)
task3 = np.array([[3, 32 if np.random.rand() < 0.1 else 0] for _ in range(16)])
while not np.any(task3 == 32):
    task3 = np.array([[3, 32 if np.random.rand() < 0.1 else 0] for _ in range(16)])
tasks.extend([task1, task2, task3])

stimuli_matrix = pd.read_excel('stimuli.xlsx', header=None).values
pattern_matrix = pd.read_excel('patterns.xlsx', header=None).values
df = pd.read_csv('data_res_by_sti_fgl_disp_hist_firstsel.csv')
selected_tasks = [9492, 14198, 17720]
task_matrices = []
for task_id in selected_tasks:
    print(int(stimuli_matrix[task_id - 1][0]), int(stimuli_matrix[task_id - 1][1]))
    left_option = pattern_matrix[int(stimuli_matrix[task_id - 1][0]-1)][:16]
    right_option = pattern_matrix[int(stimuli_matrix[task_id - 1][1]-1)][:16]
    tasks.append(np.array([[left_option[i], right_option[i]] for i in range(16)]))
    variants = df[df['index'].str.contains(f'np.int64\\({task_id}\\)')]
    print(f'Task ID: {task_id}')
    print(f'Left option pattern: {left_option}')
    print(f'Right option pattern: {right_option}')
    for _, row in variants.iterrows():
        print_variant = row['index'].replace('np.int64(', '').replace(')', '')
        print_variant = print_variant.split(', ')
        fgl = 'No FGL' if print_variant[1] == '0' else 'FGL'
        disp_type = {'0': 'Numeric', '1': 'Graphical', '2': 'Combined'}[print_variant[2]]
        history = 'No History' if print_variant[3] == '0' else 'History'
        first_sel = 'Left First' if print_variant[4] == '0' else 'Right First'
        if first_sel == 'Left First':
            print(f'Variant: Task {task_id}, {fgl}, {disp_type}, {history}, {first_sel}')
            res_mean = [f'{x:.2f}' for x in eval(row['res_mean'])]
            print(f'res_mean: {res_mean}')
    print('---')

task_names = [f'Task {i+1}' for i in range(6)]
for task_matrix, task_name in zip(tasks, task_names):
    print(f'Running simulation for {task_name}...')
    results = simulate_rw_mc(
        task_matrix, N=1000,
        partial_full=1, outcome_mode='standard'
    )
    # print(f'SIMULATING FOR {task_name} (Partial, Standard)...')
    # results_partial = simulate_rw_mc(
    #     task_matrix, N=1000,
    #     partial_full=0, outcome_mode='standard'
    # )
    # plot_rw_results(results_partial, 16, task_name, "partial-standard")
    # print(f'SIMULATING FOR {task_name} (Full, Standard)...')
    # results_full = simulate_rw_mc(
    #     task_matrix, N=1000,
    #     partial_full=1, outcome_mode='standard'
    # )
    # plot_rw_results(results_full, 16, task_name, "full-standard")

    # valid_ct_partial = [c for c in results_partial['convergence_trials'] if c != 13]
    # valid_ct_full = [c for c in results_full['convergence_trials'] if c != 13]
    # t_stat, p_val = ttest_ind(valid_ct_partial, valid_ct_full, equal_var=False)
    # print(f"Distribution of convergence trials (Partial feedback): Mean={np.mean(valid_ct_partial)+1:.2f} ({np.std(valid_ct_partial, ddof=1):.2f}, N={len(valid_ct_partial)})")
    # print(f"Distribution of convergence trials (Full feedback): Mean={np.mean(valid_ct_full)+1:.2f} ({np.std(valid_ct_full, ddof=1):.2f}, N={len(valid_ct_full)})")
    # print(f'T-test for {task_name} (Partial vs Full feedback): t={t_stat:.3f}, p={p_val:.3f}')
    # switch_count_partial = results_partial['switch_counts']
    # switch_count_full = results_full['switch_counts']
    # t_stat_sw, p_val_sw = ttest_ind(switch_count_partial, switch_count_full, equal_var=False)
    # print(f"Distribution of switch count (Partial feedback): Mean={np.mean(switch_count_partial):.2f} ({np.std(switch_count_partial, ddof=1):.2f})")
    # print(f"Distribution of switch count (Full feedback): Mean={np.mean(switch_count_full):.2f} ({np.std(switch_count_full, ddof=1):.2f})")
    # print(f'T-test for switch count for {task_name} (Partial vs Full feedback): t={t_stat_sw:.3f}, p={p_val_sw:.3f}')
    # Do the same for history-based outcome mode
    print(f'SIMULATING FOR {task_name} (Partial, History)...')
    results_partial_hist = simulate_rw_mc(
        task_matrix, N=1000,
        partial_full=0, outcome_mode='history' 
    )
    plot_rw_results(results_partial_hist, 16, task_name, "partial-history")
    print(f'SIMULATING FOR {task_name} (Partial, no-History)...')
    results_full_hist = simulate_rw_mc(
        task_matrix, N=1000,
        partial_full=0, outcome_mode='standard' 
    )
    plot_rw_results(results_full_hist, 16, task_name, "partial-no-history")
    valid_ct_partial_hist = [c for c in results_partial_hist['convergence_trials'] if c != 13]
    valid_ct_full_hist = [c for c in results_full_hist['convergence_trials'] if c != 13]
    t_stat_hist, p_val_hist = ttest_ind(valid_ct_partial_hist, valid_ct_full_hist, equal_var=False)
    print(f"Distribution of convergence trials (Partial feedback, History): Mean={np.mean(valid_ct_partial_hist)+1:.2f} ({np.std(valid_ct_partial_hist, ddof=1):.2f}, N={len(valid_ct_partial_hist)})")
    print(f"Distribution of convergence trials (Full feedback, History): Mean={np.mean(valid_ct_full_hist)+1:.2f} ({np.std(valid_ct_full_hist, ddof=1):.2f}, N={len(valid_ct_full_hist)})")
    print(f'T-test for {task_name} (Partial vs Full feedback, History): t={t_stat_hist:.3f}, p={p_val_hist:.3f}')
    switch_count_partial_hist = results_partial_hist['switch_counts']
    switch_count_full_hist = results_full_hist['switch_counts']
    t_stat_sw_hist, p_val_sw_hist = ttest_ind(switch_count_partial_hist, switch_count_full_hist, equal_var=False)
    print(f"Distribution of switch count (Partial feedback, History): Mean={np.mean(switch_count_partial_hist):.2f} ({np.std(switch_count_partial_hist, ddof=1):.2f})")
    print(f"Distribution of switch count (Full feedback, History): Mean={np.mean(switch_count_full_hist):.2f} ({np.std(switch_count_full_hist, ddof=1):.2f})")
    print(f'T-test for switch count for {task_name} (Partial vs Full feedback, History): t={t_stat_sw_hist:.3f}, p={p_val_sw_hist:.3f}') 
