Problems:

The earthquake classification problem involves predicting whether a
major event is about to occur based on the most recent readings in
the surrounding area. The data is taken from Northern California
Earthquake Data Center~\cite{ncdc} and each data is an averaged
reading for one hour, with the first reading taken on Dec 1st 1967,
the last in 2003. We transform this single time series into a
classification problem by first defining a major event as any
reading of over 5 on the Rictor scale. Major events are often
followed by aftershocks. The physics of these are well understood
and their detection is not the objective of this exercise. Hence we
consider a positive case to be one where a major event is not
preceded by another major event for at least 512 hours. To
construct a negative case, we consider instances where there is a
reading below 4 (to avoid blurring of the boundaries between major
and non major events) that is preceded by at least 20 readings in
the previous 512 hours that are non-zero (to avoid trivial negative
cases). None of the cases overlap in time (i.e. we perform a
segmentation rather than use a sliding window). Of the 86,066
hourly readings, we produce 368 negative cases and 93 positive.

Figure~\ref{earthquakeExamples} shows some examples of positive and
negative cases. The data have not been normalised.

\begin{figure}[!ht]
\centering
\begin{tabular}{cc}
\includegraphics[height= 3.5cm, width = 4cm]{Pictures/earthquakesNegative.jpg} &
\includegraphics[height= 3.5cm, width = 4cm]{Pictures/earthquakesPositive.jpg}
 \end{tabular}

\caption{\scriptsize Examples of ten negative (left) and ten positive (right) cases in the earthquakes data set.}
\label{earthquakeExamples}
\end{figure}
