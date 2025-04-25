# imc-prosperity3 by the Tropical TyGoon

## Contributors
**Zain Al-Saffi**

**Masham Siddiqui**

**Zac Kienzle**

**Roger Zhu**

**Abhinav Pradeep**


### Results
| Round | Overall Position | Manual      | Algorithmic | Country |
|-------|------------------|-------------|-------------|---------|
| 1     | 268              | Tied 1st    | 300         | 10th    |
| 2     | 467              | 966         | 516         | 23rd    |
| 3     | 67               | 732         | 61          | 4th     |
| 4     | 54               | 90          | 69          | 4th     |
| 5     | 89               | 626         | 60          | 9th     |





#### Round 1

3 assets were introduced, `SQUID_INK`, `RAINFOREST_RESIN`, and `KELP`. These assets each exhibited different behaviours in the simulation. We found Rainforest to be highly stationary and hovered around a mean of 10,000. We implemented a market-make and market-take strategy around this mean based on a GLFT, we also tested an Ornstein-Uhlenbeck process to model the price of Rainforest. We found that the GLFT was more effective in capturing the mean reversion of the asset. SQUID INK was highly volatile and we did not quit figure it out this round, however later on we discovered that if you find the mean reverting spread within squink and run an EMA on it, you can generate stable signals on the EMA due to the large spikes in the price. Kelp was a bit more difficult to model, we found that it was highly correlated with Rainforest and SQUID INK, however it was not stationary. We used a simple market-making strategy on Kelp and it worked well. Zac and Abhinav traded the manual strategy, in which you can solve the matrix problem and find the optimal conversions. Hence this resulted in our tied 1st place with 900 other people as there was a clear cut solution to the manual strategy.


#### Round 2

#### Round 3

#### Round 4

#### Round 5
