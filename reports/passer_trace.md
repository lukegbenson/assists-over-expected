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
- training agreement at the chosen combination: 94.25% (false-NONE rate 3.65%)

## Held-out accuracy

- assisted makes evaluated: 7,030
- **agreement with the known assister: 94.67%**
- **false-NONE rate (assisted makes traced as NONE): 3.27%**

## Overall (every included make, assisted or not)

- makes with a traced passer attempt: 39,506
- share traced as NONE: 15.55%
- share where the run active at the shot wasn't the recorded shooter's (trace_problem): 7.56%

## Example disagreements (assisted makes; traced passer vs. the known assister)

| game_id | eventnum | description | assister_id | traced_passer_id | trace_problem | set |
|---|---|---|---|---|---|---|
| 0021500083 | 52 | Barnes  Dunk (9 PTS) (Thompson 1 AST) | 202691 | 201939 | True | train |
| 0021500063 | 505 | Williams 24' 3PT Jump Shot (22 PTS) (Thompson 1 AST) | 202684 |  | False | train |
| 0021500623 | 521 | Crowder 2' Running Layup (9 PTS) (Bradley 5 AST) | 202340 |  | False | holdout |
| 0021500246 | 464 | Batum 25' 3PT Jump Shot (15 PTS) (Lamb 2 AST) | 203087 | 101107 | False | train |
| 0021500191 | 289 | James  Running Layup (7 PTS) (Dellavedova 2 AST) | 203521 |  | False | train |
| 0021500339 | 325 | Gordon 26' 3PT Jump Shot (13 PTS) (Holiday 3 AST) | 201950 | 201936 | True | train |
| 0021500273 | 486 | Hickson  Dunk (5 PTS) (Mudiay 7 AST) | 1626144 | 202702 | True | train |
| 0021500415 | 136 | Thornton 24' 3PT Jump Shot (3 PTS) (Motiejunas 2 AST) | 202700 |  | False | train |
| 0021500334 | 363 | Hezonja 26' 3PT Jump Shot (6 PTS) (Gordon 2 AST) | 203932 |  | False | train |
| 0021500126 | 444 | Crawford 25' 3PT Jump Shot (18 PTS) (Prigioni 3 AST) | 203143 |  | False | train |
| 0021500352 | 124 | McCollum 18' Pullup Jump Shot (9 PTS) (Leonard 1 AST) | 203086 | 202334 | False | holdout |
| 0021500339 | 336 | Davis 17' Jump Shot (14 PTS) (Holiday 5 AST) | 201950 |  | False | train |
| 0021500124 | 423 | Winslow 1' Layup (6 PTS) (Bosh 4 AST) | 2547 | 2757 | False | holdout |
| 0021500061 | 322 | Ginobili 3' Driving Layup (7 PTS) (Mills 1 AST) | 201988 | 200746 | False | holdout |
| 0021500103 | 289 | Leonard  Dunk (15 PTS) (Duncan 1 AST) | 1495 |  | False | train |
| 0021500058 | 591 | McLemore  Dunk (2 PTS) (Cauley-Stein 1 AST) | 1626161 |  | False | train |
| 0021500136 | 22 | Gallinari 24' 3PT Jump Shot (3 PTS) (Mudiay 2 AST) | 1626144 | 201581 | False | train |
| 0021500184 | 372 | Hayward  Running Dunk (16 PTS) (Burke 2 AST) | 203504 |  | False | train |
| 0021500137 | 240 | Robinson 2' Cutting Layup Shot (2 PTS) (Jack 5 AST) | 101127 | 202711 | True | train |
| 0021500517 | 506 | Collison 2' Layup (14 PTS) (Rondo 17 AST) | 200765 |  | False | holdout |
