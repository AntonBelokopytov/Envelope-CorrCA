Function to find comodulated induced activity across multiple trials

## Component example

![Component example](docs/images/comp_example.png)


\section*{Envelope-CorrCA}

Envelope-CorrCA extracts spatial components whose induced (power) activity 
is maximally correlated across trials. The method operates in the space of 
covariance matrices computed from short sliding windows.

\subsection*{Data}

Let
\[
X \in \mathbb{R}^{T \times C \times N}
\]
where
\begin{itemize}
\item $T$ -- number of time samples,
\item $C$ -- number of channels,
\item $N$ -- number of trials.
\end{itemize}

\subsection*{Epoching}

Each trial is segmented into overlapping windows of length $W$ with step $S$:
\[
X^{(n)} \rightarrow X^{(n)}_e, 
\qquad e = 1,\dots,E.
\]

\subsection*{Covariance Time Series}

For each window:
\[
C^{(n)}_e = \operatorname{cov}(X^{(n)}_e).
\]

Mean covariance:
\[
C_{xx} = \frac{1}{E} \sum_{e=1}^{E} C_e.
\]

\subsection*{Whitening}

Regularized covariance:
\[
C_{xx}^{reg} = C_{xx} + \lambda I.
\]

Whitening matrix:
\[
W_m = C_{xx}^{-1/2}.
\]

Whitened covariance:
\[
\tilde{C}^{(n)}_e = W_m C^{(n)}_e W_m^\top.
\]

\subsection*{Component Power}

For a spatial filter $w$, window power is defined as
\[
p^{(n)}_e = w^\top C^{(n)}_e w.
\]

\subsection*{Optimization Objective}

Envelope-CorrCA maximizes inter-trial correlation of power:
\[
\rho = 
\text{mean off-diagonal}
\left(
\operatorname{corr}
\left(
p^{(1)}_e, \dots, p^{(N)}_e
\right)
\right).
\]

\subsection*{Light Version}

The problem reduces to a generalized eigenvalue problem:
\[
R_b w = \lambda R_w w,
\]
where
\[
R_w = \mathbb{E}[C C^\top], 
\qquad
R_b = \mathbb{E}[C_i C_j].
\]

\subsection*{Full Version}

\begin{enumerate}
\item Vectorize upper triangular parts of covariance matrices.
\item Apply PCA for dimensionality reduction.
\item Apply CorrCA in covariance space:
\[
R_b v = \lambda R_w v.
\]
\end{enumerate}

\subsection*{Normalization}

Spatial filters are normalized as
\[
w^\top C_{xx} w = 1.
\]

Spatial patterns are computed as
\[
a = C_{xx} w.
\]