# Pass attribution, 2015-16

## Chosen parameters (grid search, maximizing agreement with the known
assister on a training sample of assisted makes; ties broken toward the
lower false-NONE rate -- see config.yaml's passer: section)

- hold_dist: 4.00 ft
- hold_height: 9.00 ft
- min_run_s: 0.20 s
- dribble_gap_s: 0.80 s

## Tuning

- training games: 429; held-out games: 184
- training sample (assisted makes): 2,000
- training agreement at the chosen combination: 94.35% (false-NONE rate 3.35%)

## Held-out accuracy

- assisted makes evaluated: 6,854
- **agreement with the known assister: 94.76%**
- **false-NONE rate (assisted makes traced as NONE): 3.22%**

## Overall (every included make, assisted or not)

- makes with a traced passer attempt: 38,666
- share traced as NONE: 16.44%
- share where the run active at the shot wasn't the recorded shooter's (trace_problem): 7.59%

## Example disagreements (assisted makes; traced passer vs. the known assister)

| game_id | eventnum | description | assister_id | traced_passer_id | trace_problem | set |
|---|---|---|---|---|---|---|
| 0021500082 | 211 | Mark Morris 8' Turnaround Bank Shot (4 PTS) (Bledsoe 2 AST) | 202339 | 202688 | False | holdout |
| 0021500283 | 145 | Grant 1' Layup (2 PTS) (Galloway 1 AST) | 204038 |  | False | holdout |
| 0021500183 | 38 | Waiters 2' Driving Layup (4 PTS) (Ibaka 1 AST) | 201586 | 201566 | False | train |
| 0021500236 | 344 | Curry 24' 3PT Jump Shot (36 PTS) (Iguodala 2 AST) | 2738 |  | True | train |
| 0021500487 | 119 | W. Johnson 24' 3PT Jump Shot (3 PTS) (Prigioni 1 AST) | 203143 |  | True | holdout |
| 0021500328 | 378 | Bazemore 1' Driving Layup (9 PTS) (Schroder 1 AST) | 203471 |  | True | train |
| 0021500661 | 299 | Harris 2' Driving Layup (8 PTS) (Arthur 3 AST) | 201589 |  | False | holdout |
| 0021500201 | 105 | Russell 25' 3PT Jump Shot (5 PTS) (Bass 1 AST) | 101138 | 101150 | True | train |
| 0021500157 | 190 | Warren  Running Dunk (3 PTS) (Knight 3 AST) | 202688 |  | False | train |
| 0021500320 | 188 | Walker 1' Running Layup (7 PTS) (Batum 7 AST) | 201587 |  | False | train |
| 0021500341 | 346 | Booker  Running Dunk (6 PTS) (Hayward 2 AST) | 202330 | 203481 | True | train |
| 0021500435 | 504 | Brown 2' Running Layup (2 PTS) (Huertas 4 AST) | 1626273 |  | False | train |
| 0021500310 | 281 | Plumlee  Cutting Dunk Shot (3 PTS) (Lillard 5 AST) | 203081 | 203468 | False | train |
| 0021500064 | 402 | Ariza 24' 3PT Jump Shot (17 PTS) (Terry 2 AST) | 1891 | 201951 | True | train |
| 0021500479 | 392 | Hayward 25' 3PT Jump Shot (12 PTS) (Booker 2 AST) | 202344 |  | True | train |
| 0021500029 | 24 | Harris 2' Reverse Layup (2 PTS) (Gallinari 1 AST) | 201568 | 203530 | False | train |
| 0021500350 | 131 | Oubre Jr. 6' Jump Shot (3 PTS) (Dudley 2 AST) | 201162 |  | False | holdout |
| 0021500220 | 197 | Ariza 1' Driving Layup (6 PTS) (Harden 2 AST) | 201935 | 2730 | False | train |
| 0021500474 | 409 | Prigioni 26' 3PT Jump Shot (3 PTS) (W. Johnson 1 AST) | 202325 | 2037 | False | train |
| 0021500531 | 2 | Gordon 26' 3PT Jump Shot (3 PTS) (Davis 1 AST) | 203076 | 201936 | False | train |
