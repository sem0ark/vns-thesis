From https://github.com/vOptSolver/vOptLib/blob/master/UKP/uncorrelated.md

The instances are denoted by `2KPn-1B.dat` where n is the size of the problem.
- 1st objective cost and the weights are generated according to a uniform distribution in [1,...,100].
- 2nd objective is obtained by taking the objective cost of the first one in reverse order.

All instances have a tightness ratio `W / (Sum{i=1,...n} w(i)) = 0.5`.
There are 10 instances:
- `2KP50-1B.dat`
- `2KP100-1B.dat`
- `2KP150-1B.dat`
- `2KP200-1B.dat`
- `2KP250-1B.dat`
- `2KP300-1B.dat`
- `2KP350-1B.dat`
- `2KP400-1B.dat`
- `2KP450-1B.dat`
- `2KP500-1B.dat`


```
Max sum{i=1,…,n} c1(i) x(i)
Max sum{i=1,…,n} c2(i) x(i)
s/t sum{i=1,…,n} w(i)  x(i) <= W
                 x(i)=0 or 1        for i=1,...,n
```

File format:
```json
{
    "metadata": {
        "objectives": 2,
        "weights": 1,
        "capacity": 0
    },
    "weights": [],
    "objectives": [
        [],
        []
    ],
    "reference_front": []
}
```
