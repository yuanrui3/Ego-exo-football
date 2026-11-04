# Causal feature contracts

Every model input ends at query q or earlier. Native25Hz indices remain distinct from event displayed game time. There are no match IDs, player IDs, source annotation indices, future coordinates, or future visibility masks in feature matrices. IDs are used privately for joins/folds only.

## Compact state12

0 availability of matched current player and as-of ball;1 ball age /0.2s;2/3 ball-minus-player XY /105m,68m;4 distance /105m;5 player approach /12m/s;6 ball speed /40m/s;7 relative closing /40m/s;8/9 player absoluteXY /52.5m,34m;10/11 backward0.2s player velocity /12m/s. Ball velocity is backward native as-of difference, never a centred derivative. Entire unavailable states are zero with availability0; coordinates are not silently interpolated.

## Dynamics14 and current competition9

Six snapshots at q-50,-40,-30,-20,-10,q frames supply availability fraction,2s and0.4s distance change,2s XY change, ball-speed change, defined relative approach/TCA, closest approach, historical minimum distance and2/5m occupancy, and relative velocityXY. Defined flags guard zero-speed cases. TCA clips to0-3s. Current candidate competition adds nearest distance/gap, stable rank, distance-softmax weight, close-player counts, distance margin, entropy and availability fraction. No future target is consulted.

Compact kinematics = state12 + dynamics14 + competition9 =35channels. Raw history = state12 + five older8channel states =52. Full inherited interpretable model =196 original verified causal actor features +12 ball context +14 dynamics +9 competition =231; it has richer inputs and is not an equal-information compact architecture comparison. Time-to-contact model appends6 fixed-horizon projection features to35.

## Source-team context18

Same team5 and opposing team5: nearest distance /105m; counts within3,5,10m /11; nearest closing speed /12m/s. Candidate-to-ball corridor adds minimum transverse separation /68m and count within2m /11 separately for same and opposing teams. Four remaining channels: team known, absolute normalized fieldXY, current-roster fraction with known team. Missing corridor/ball data are zero and paired with the original ball-availability state. No-opponent distance sentinel1000m is retained and clipped consistently. No privileged possession or guessed attack direction is supplied.

Same/opponent ablations each append their5 pressure +2 corridor channels and2 quality indicators to35. Full context appends all18. These are source-observed **exocentric current roster** features, not simulated human awareness or visibility.

## Counterfactuals and conditional tasks

The0.48s stale arm reconstructs every compact state and historical snapshot at q-12 or earlier for both player and ball; it does not combine old ball with current actor. Candidate identity is fixed at q, so this is stale information for a current-candidate query, not an online roster generator.

The adaptive smoother retains the35channel layout while replacing actor/ball finite differences with1s backward changes and relative approach with0.8s backward changes. It never uses future interpolation. Missing long-baseline endpoints zero their derivative and retain validity/context flags.

Action-type models and GPU scene-softmax models are explicitly conditional on a supported future occurrence for diagnostic ranking/type evaluation. They are not occurrence detectors. Unknown source actors, ambiguous simultaneous next actors, geometry ties<=0.2m and incomplete future labels are censored as target-only conditions, not fed to models.

Importance uses held-match grouped permutations and one-feature partial dependence. These may create physically impossible combinations in correlated feature groups; their AP drops are model-sensitivity summaries, not causal tactical effects.
