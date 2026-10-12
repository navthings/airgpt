<div align="center">

<h1>airgpt</h1>

<p>train your own chatbot from scratch on a mac</p>

<p>
<a href="#quick-start">quick start</a> &nbsp;|&nbsp;
<a href="#what-an-hour-gets-you">results</a> &nbsp;|&nbsp;
<a href="#how-fast-is-your-mac">bench</a> &nbsp;|&nbsp;
<a href="#speedrun">speedrun</a>
</p>

<img src="assets/story.gif" width="760" alt="the hour model writing a story in the terminal">

<p><sub>a model trained for one hour on a macbook air, writing a story. real speed.</sub></p>

</div>

<br>

one command downloads some data, trains a tokenizer, pretrains a small llama-style model, teaches it to chat, and lets you talk to it. it runs on any apple silicon mac, fanless air included, for exactly as long as you tell it to. no cuda, no cloud, just [mlx](https://github.com/ml-explore/mlx).

i made this because every "train your own llm" repo i found wanted an nvidia gpu, and i have a macbook air.

## quick start

```bash
git clone https://github.com/navthings/airgpt && cd airgpt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py --preset smoke   # 3 min, checks everything works
python train.py --preset hour    # the real thing
python chat.py runs/hour
```

| preset | params | time | what you get |
|---|---|---|---|
| `smoke` | 0.7M | 3 min | proof it works |
| `hour` | 8M | 1 hour | little stories, says hi, makes up every fact |
| `night` | 40M | 8 hours | <!-- night summary --> a bigger model, results coming |

`--minutes 30` changes the time. 80% goes to pretraining and 20% to chat.

pick what it learns from with `--mix`:

```bash
python train.py --preset hour --mix fineweb-edu
python train.py --preset hour --mix tinystories=0.7,wikipedia=0.3
```

built in: `tinystories`, `cosmopedia`, `fineweb-edu`, `wikipedia`, `gutenberg`, or any hugging face dataset with a `text` column (`--mix someone/their-dataset`).

got an 8gb mac? it checks your ram before it starts, and if a preset won't fit it runs each batch in smaller pieces and adds them up. same training, just slower.

## what an hour gets you

on my macbook air (m5, 16gb) the `hour` preset trained an 8M parameter model on 55M tokens. every two minutes it samples the model with the same prompt, so you can watch it learn to write:

| | val loss | "Once upon a time" |
|---|---|---|
| 2 min | 4.09 | , there was a little boy named Tim. Timmy loved to play with his friends. One day, Timmy was a very excited. |
| 6 min | 3.15 | , there was a little girl named Lily. She had a big red dress that she loved to play with. One day, Lily went to a party with her mom. |
| 46 min | 2.27 | , there was a little girl named Lily. She loved to play with her toys and her favorite toy was a big, red ball. One day, Lily's mom asked her to help |

<img src="assets/hour.png" width="100%" alt="loss and speed over the hour run">

it never got hot. the thermal state stayed at nominal for the whole hour and speed sat around 20k tokens/s. the dips are where it stops to evaluate and write a sample.

the last 12 minutes teach it to chat. at 8M params it picks up the shape of a conversation and not much else. first tries, unedited:

```
> hi!
Hello! How can I help you today?

> what is the capital of france?
The capital of france is the capital of the france of the tech of decimal.

> how do i make friends at a new school?
Certainly! i made sense to the school students in the classroom. The school students were interested in learning and learning, and then they would love to discuss each other's work and make a team of friends.
```

<!-- night -->

## how fast is your mac?

```bash
python bench.py
```

two minutes, nothing to download. it trains the `hour` model on random tokens and prints your speed, how much it dropped by the end, and a row for this table. open a pr with yours.

| mac | chip | memory | preset | tok/s | after 2 min | thermal |
|---|---|---|---|---|---|---|
<!-- bench rows -->

## speedrun

lowest val loss from one hour of training, with records kept per chip. the m5 air number is the one to beat, and if nobody's posted your chip yet, any run is a record.

| chip | val loss | who | what changed |
|---|---|---|---|
| m5 (macbook air, 16gb) | 2.265 | [@navthings](https://github.com/navthings) | baseline |

<details>
<summary><b>rules</b></summary>
<br>

- `python train.py --preset hour` on the `data/hour` that `prepare.py` makes. don't change `prepare.py` or the eval.
- 48 minutes of pretraining on train.py's clock, same as the preset. data prep doesn't count.
- everything else is fair game: the model, the optimizer, the schedule, the batch size.
- the score is `pretrain.val_loss` in `runs/hour/summary.json`.
- open a pr adding your row, with your `summary.json` and `log.jsonl` and a sentence on what you changed.

</details>

## more

<details>
<summary><b>what it's actually doing</b></summary>
<br>

first it streams text from [tinystories](https://huggingface.co/datasets/roneneldan/TinyStories) and [cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus). tinystories is simple english written for small models, which is most of the reason an 8M param model can write real sentences after a few minutes.

then it trains its own byte-level bpe tokenizer (`tokenizer.py`). it starts with the 256 possible bytes and keeps merging the most common pair until it has 8192 tokens. no tiktoken, no hugging face tokenizers.

pretraining is the model reading that text and guessing the next token over and over. that's where it learns english.

chat training comes after, on short conversations from [everyday-conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) and [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) laid out like `<|user|> hi <|assistant|> hey! <|end|>`. the loss only counts the assistant's tokens, so it learns to answer you instead of copying you.

the learning rate runs off the clock instead of a step count: warmup for the first 3% of the time, then cosine down to 10%. a slow mac sees fewer tokens than a fast one, but both finish on time with the schedule fully decayed.

the first run of each preset spends a few minutes downloading and tokenizing before training starts. after that it reuses `data/<preset>`.

</details>

<details>
<summary><b>train it on your own stuff</b></summary>
<br>

```bash
python train.py --preset hour --text ~/notes
python train.py --preset hour --text book.txt --chat my_chats.jsonl
```

`--text` takes a file or a folder of .txt/.md files. with only that you get a model that continues text in the style of your files (`python chat.py runs/hour --base`). add `--chat` and it learns to reply too. one conversation per line:

```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey!"}]}
```

</details>

<details>
<summary><b>stuff to know</b></summary>
<br>

macbook airs have no fan. the trainer reads macos's thermal state, shows it while it trains and logs it, and if it ever hits critical it saves and pauses for 30 seconds. `plot.py` marks any hot stretches in the speed chart.

it makes things up. 8M params is enough to learn english, not facts.

sampling is cold on purpose (temperature 0.6, min-p 0.1). small models need colder sampling than big ones, which is what my current paper is about. try `--temp 0.3` and `--temp 1.0` to see the difference.

all three presets fit in 16gb without splitting (the hour run peaked at 5.8gb). on 8gb the hour and night presets get split in two.

it keeps the mac awake while it trains, but closing the lid still puts it to sleep, so leave it open and plugged in.

ctrl-c saves and prints the exact command to carry on, clock included. it also checkpoints every 10 minutes. `--restart` throws a run away and starts over.

`python plot.py runs/hour` saves loss and speed over time to `runs/hour/plot.png`.

</details>

<details>
<summary><b>files</b></summary>
<br>

| file | |
|---|---|
| `train.py` | training loop, clock-based schedule, logging, checkpoints |
| `model.py` | the transformer: rmsnorm, rope, grouped-query attention, swiglu, tied embeddings, kv cache |
| `tokenizer.py` | byte-level bpe |
| `prepare.py` | downloads, tokenizes and packs the data |
| `chat.py` | talk to it |
| `bench.py` | 2 minute speed test |
| `config.py` | presets |
| `mac.py` | chip, memory, fanless check, thermal state |
| `plot.py` | loss and speed charts |
| `ui.py` | the terminal output |

</details>

## credits

[nanogpt](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat) and [minimind](https://github.com/jingyaogong/minimind) for the idea. tinystories (eldan & li) and hugging face's smollm datasets for the data. apple's mlx for making this possible on a laptop.

mit license
