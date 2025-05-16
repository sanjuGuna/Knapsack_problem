import streamlit as st
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def knapsack_bottom_up(items, capacity):
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if items[i-1]['weight'] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    items[i-1]['value'] + dp[i-1][w - items[i-1]['weight']]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)
            w -= items[i-1]['weight']
    
    return dp, dp[n][capacity], selected[::-1]

def knapsack_top_down(items, capacity):
    n = len(items)
    memo = [[-1] * (capacity + 1) for _ in range(n + 1)]
    
    def recurse(i, w):
        if i == 0 or w == 0:
            return 0
        if memo[i][w] != -1:
            return memo[i][w]
        if items[i-1]['weight'] > w:
            memo[i][w] = recurse(i-1, w)
        else:
            memo[i][w] = max(
                recurse(i-1, w),
                items[i-1]['value'] + recurse(i-1, w - items[i-1]['weight'])
            )
        return memo[i][w]
    
    optimal_value = recurse(n, capacity)
    return memo, optimal_value

def plot_dp_table(dp, title, figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    data = np.array(dp)
    
    # heatmap with clearer colors
    cax = ax.matshow(data, cmap='YlGnBu')
    fig.colorbar(cax, label='Max Value')
    
    
    ax.set_xlabel('Capacity →', fontsize=10)
    ax.set_ylabel('Items ↓', fontsize=10)
    ax.set_title(title, fontsize=12, pad=20)
    
    #table  anotations
    if len(dp) <= 10 and len(dp[0]) <= 10:
        for i in range(len(dp)):
            for j in range(len(dp[0])):
                ax.text(j, i, f"{dp[i][j]}", 
                    ha='center', va='center', 
                    color='black' if dp[i][j] < max(map(max, dp))/2 else 'white',
                    fontsize=8)
    
    st.pyplot(fig)

def main():
    st.set_page_config(layout="centered")
    st.title("0/1 Knapsack Problem Solver")
    st.markdown("""
    Compare the **Bottom-Up (Tabulation)** and **Top-Down (Memoization)** approaches.
    Enter your problem details below:
    """)
    
    #Input
    st.header("Problem Setup")
    
    capacity = st.number_input("Knapsack Capacity", min_value=1, value=10, step=1)
    
    st.subheader("Add Items")
    num_items = st.number_input("Number of Items", min_value=1, max_value=10, value=4, step=1)
    items = []
    cols = st.columns(2)
    for i in range(num_items):
        with cols[0]:
            weight = st.number_input(f"Weight of Item {i+1}", min_value=1, value=2 if i==0 else 1, key=f"w{i}")
        with cols[1]:
            value = st.number_input(f"Value of Item {i+1}", min_value=1, value=300 if i==0 else 200, key=f"v{i}")
        items.append({'weight': weight, 'value': value})
    
    if st.button("Solve Knapsack Problem"):
        # Bottom-Up
        start_bu = time.perf_counter()
        dp, optimal_value, selected = knapsack_bottom_up(items, capacity)
        time_bu = (time.perf_counter() - start_bu) *1000
        
        # Top-Down
        start_td = time.perf_counter()
        memo, optimal_td = knapsack_top_down(items, capacity)
        time_td = (time.perf_counter() - start_td) * 1000
        
        #results
        st.success(f"**Optimal Value:** {optimal_value}")
        
        # Selected Items
        st.subheader("Selected Items")
        selected_items = [items[i] for i in selected]
        df = pd.DataFrame({
            'Item': [f"Item {i+1}" for i in selected],
            'Weight': [item['weight'] for item in selected_items],
            'Value': [f"{item['value']}" for item in selected_items]
        })
        st.table(df)
        
        #comparison
        st.subheader("Comparison of Approaches")
        st.header("Approach Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Bottom-Up (Tabulation)")
            st.markdown(f"**Time:** {time_bu:.2f} ms")
            st.markdown("""
            - Iterative approach
            - Builds solution from base cases up
            - Always computes all subproblems
            - Generally faster for small problems
            """)
        
        with col2:
            st.markdown("### Top-Down (Memoization)")
            st.markdown(f"**Time:** {time_td:.2f} ms")
            st.markdown("""
            - Recursive approach
            - Only computes needed subproblems
            - More intuitive implementation
            - Better for problems with sparse DP table
            """)
        
        # Visualizations
        st.header("DP Table Visualizations")
        
        tab1, tab2 = st.tabs(["Bottom-Up Table", "Top-Down Table"])
        
        with tab1:
            plot_dp_table(dp, "Bottom-Up DP Table", figsize=(8, 5))
        
        with tab2:
            plot_dp_table(memo, "Top-Down DP Table", figsize=(8, 5))

if __name__ == "__main__":
    main()