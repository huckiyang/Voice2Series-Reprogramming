Probems Worms and WormsTwoClass

Background:
 {\em Caenorhabditis elegans} is a roundworm commonly
used as a model organism in the study of genetics. The movement of
these worms is known to be a useful indicator for understanding
behavioural genetics. Brown {\em et al.}~\cite{brown13worms}
describe a system for recording the motion of worms on an agar
plate and measuring a range of human-defined
features~\cite{yemini13wormsdatabase}. It has been shown that the
space of shapes {\em Caenorhabditis elegans} adopts on an agar
plate can be represented by combinations of four base shapes, or
\emph{eigenworms}. Once the worm outline is extracted, each frame
of worm motion can be captured by four scalars representing the
amplitudes along each dimension when the shape is projected onto
the four eigenworms (see Figure~\ref{wormFig}). Using data
collected for the work described in~\cite{brown13worms},

Data Processing:

The data relates to 258 traces of worms converted into four
"eigenworm" series. The eigenworm data are lengths from 17984 to
100674 (sampled at 30 Hz, so from 10 minutes to 1 hour) and in four
dimensions (eigwnworm 1 to 4). There are five
classes:N2,goa-1,unc-1,unc-38 and un63. N2 is wildtype (i.e.
normal) the other 4 are mutant strains.

The problems Worms.arff and WormsTwoClass.arff are series of first
eigenworm1 averaged down so that all series are lengths 900 (the
single hour long series is discarded). This smoothing is likely to
discard discriminatory information. The Yemini features obtains
nearly 100\% accuracy, although we have not independently verified
this. 


we address the problem of classifying individual worms as wild-type
or mutant based on the time series of the first eigenworm,
down-sampled to second-long intervals. We have 257 cases, which we
split 70\%/30\% into a train and test set. Each series has 900
observations, and each worm is classified as either wild-type (the
N2 reference strain - 109 cases) or one of four mutant types: goa-1
(44 cases); unc-1 (35 cases); unc-38 (45 cases) and unc-63 (25
cases). The data were extracted from the {\em C. elegans}
behavioural database~\cite{wormWeb}. The formatted classification
problems are available from the website associated with this
paper~\cite{tscWeb}.
