# Goal 

Do parameter sweep described in chat history.

# Chat history

Chat history with github copilot is `/home/irafm/FRNN/chat_summary.md`.

# Compute 

Use gpu using pytorch in env1 conda enviroment. Do not modify the enviroment. If you need to do so, notify the user.

# Success criterion

- Run sweep for f = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20}
- Each run is for 100 epochs
- Verify that python code in this project matches the matlab code. Fix any deviation.
- Produce latex table according to example below.
- Dump the results to csv too, for number verifaction.
- Use tmux so that user can disconnect from ssh.

# Table example

\begin{table}[htbp]
\centering
\caption{Training parameters and performance ....
Batch size = 1000, 60 batches, 50 epochs.}
\label{}
\begin{tabular}{cccccc}
\toprule
$f$ & Num. Neuron & Loss    & Train acc & Test acc & $T$ (s) \\
\midrule
2      & 276  & 0.1555 & 84.5583 &84.5700 & 9.3530  \\
\bottomrule
\end{tabular}%
\end{table}