# Attention Is All You Need

PyTorch implementations of [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) and [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

The architecture of Attention Is All You Need can be found in [Optimus Prime](GPT1/model/optimus_prime/) and the architecture of GPT-1 in [Bumblebee](GPT1/model/bumblebee/)

Implementation details were kept as similar to those in Attention Is All You Need as possible:

- $d_{model} = 512$
- $h = 8$ attention heads
- $d_k = d_v = d_{model} / h = 64$
- 6 encoder + decoder layers (Attention Is All You Need) / 12 decoder layers (GPT-1)
- Postnorm was used in favor of prenorm

<img src="https://miro.medium.com/v2/1*f3L_gGaNy9wVuTenyQJLEA.png" height="400">

## Wikitext-103

The model was trained on next-token prediction using the Wikitext-103 dataset. We used bytepair encoding (BPE) with a vocabulary size of 30,000. The dataset was split into batches of non-overlapping sentences each of length 128 tokens. 

## Results

We trained our model for 2 epochs and achieved a perplexity score of 55.9

### Example prompt

Input: `Lionel Andrés Messi ( born 24 June 1987 ) is an Argentine professional footballer who plays as a forward for and captains both the Major League Soccer club Inter Miami and the Argentina national team . Widely known as the `

Output: `greatest footballer in football and international football , he was predominantly a free agent who had a strong reputation as a top scorer . He signed for promotion to the Premier League in 1987 , and spent two seasons with the club. He was re @-@ signed by League Two club Hammond , where he scored twice in eight appearances and scored twice in the club 's first ever league game . He signed a new one @-@ year contract with the club in July 2002 , which resulted in the club being relegated back to League Two on a month later . He signed for League Two side York City in December 2002 .`

