# Skidpad relocalization under map errors

Output of `python tools/skidpad_quality.py --trials 12` on 27 Nov 2026, at commit f3066b0.
Each row is 12 replays of the bundled skidpad run with the cone map degraded:
position errors that stick to each cone (noise, metres), cones missed for the
whole run (miss), and false detections redrawn each frame (false). "wrong" counts
relocalizations whose transform puts the skidpad path more than 0.5 m from where
the clean run puts it; "error" is that distance.

```
noise m  miss false  variant             relocalized wrong frame p50 error p50 m error max m ms/frame
   0.00   0.0     0  triples (default)      12/12        0        16       0.000       0.000      1.4
   0.00   0.0     0  sampled 4-5            12/12        1        16       0.000      35.436      4.1
   0.00   0.0     0  exhaustive 4-5         12/12        0        16       0.000       0.000      4.3
   0.00   0.0     0  triples + start        12/12        0        16       0.000       0.000      1.4
   0.00   0.0     0  sampled + start        12/12        0        16       0.000       0.000      4.3

   0.05   0.0     0  triples (default)      12/12       12        14      15.201      41.900      0.9
   0.05   0.0     0  sampled 4-5            12/12       12        12      15.003      41.900      0.9
   0.05   0.0     0  exhaustive 4-5         12/12       11        12      15.194      42.121      1.0
   0.05   0.0     0  triples + start        12/12       10        18       0.693       3.045      2.4
   0.05   0.0     0  sampled + start        12/12        9        18       1.087       9.061      7.8

   0.10   0.1     0  triples (default)      12/12       11        17       4.226      15.368      0.7
   0.10   0.1     0  sampled 4-5            12/12       12        16      15.009      41.178      1.2
   0.10   0.1     0  exhaustive 4-5         12/12       11        16      11.773      33.531      1.2
   0.10   0.1     0  triples + start        11/12        7        21       0.769       6.919      3.3
   0.10   0.1     0  sampled + start        12/12        9        22       1.043       6.919      8.7

   0.20   0.2     1  triples (default)      12/12       12        28      16.344      41.748      4.8
   0.20   0.2     1  sampled 4-5            12/12       12        25      20.967      37.608     16.3
   0.20   0.2     1  exhaustive 4-5         12/12       12        22      20.184      42.260      9.0
   0.20   0.2     1  triples + start        11/12       11        60       2.446       7.456     24.3
   0.20   0.2     1  sampled + start        12/12       11        28       6.275       8.156     16.0

   0.30   0.3     2  triples (default)      12/12       12        46      14.374      42.606     14.5
   0.30   0.3     2  sampled 4-5            12/12       12        35      18.349      42.606     20.0
   0.30   0.3     2  exhaustive 4-5         12/12       12        28      22.423      43.160     19.8
   0.30   0.3     2  triples + start         9/12        9        99       4.657       9.387     24.2
   0.30   0.3     2  sampled + start        10/12       10        64       6.427       9.341     35.4


```

What it shows:

- On clean data every variant relocalizes on frame 16 onto the same transform,
  except one seed of the sampled 4-5 variant, which relocalized 35 m off.
- With 5 cm of map noise every variant without the start check is wrong in
  nearly every trial, by 10-40 m: circles fitted through entry-lane cones pass
  the centre-distance test before the real circles are in view.
- Larger subsets, sampled or exhaustive, do not reduce that, and cost about
  three times the time per frame before relocalization.
- The start check removes the gross errors (max error 3-9 m instead of
  40 m) but the transform is still off by about 1 m at 5 cm of noise, and
  relocalization comes later or not at all as the map gets worse.
