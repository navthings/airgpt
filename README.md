# airgpt

train your own chatbot from scratch on a mac.

```bash
python train.py --preset hour
```

that downloads some data, trains a tokenizer, pretrains a tiny llama-style model, teaches it to chat, then lets you talk to it. while it trains it shows a sample from the model every few minutes, so you can watch it go from gibberish to stories. it runs for however long you tell it to, so the `hour` preset takes an hour whether you're on a macbook air or a mac studio. no cuda, no cloud, nothing on top of [mlx](https://github.com/ml-explore/mlx).

i made this because every "train your own llm" repo i found wanted an nvidia gpu, and i have a fanless macbook air.

<!-- plot from a real run goes here -->

## run it

```bash
git clone https://github.com/navthings/airgpt
cd airgpt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py --preset smoke   # 3 min, checks everything works
python train.py --preset hour
python chat.py runs/hour
```

## what it's actually doing

first it streams text from [tinystories](https://huggingface.co/datasets/roneneldan/TinyStories) and [cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus). tinystories is simple english written for small models, which is most of the reason an 8M param model can write real sentences after an hour.

then it trains its own byte-level bpe tokenizer (`tokenizer.py`). it starts with the 256 possible bytes and keeps merging the most common pair until it has 8192 tokens. no tiktoken, no hugging face tokenizers.

pretraining is the model reading that text and guessing the next token over and over. that's where it learns english.

chat training comes after, on short conversations from [everyday-conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) and [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) laid out like `<|user|> hi <|assistant|> hey! <|end|>`. the loss only counts the assistant's tokens, so it learns to answer you instead of copying you.

the learning rate runs off the clock instead of a step count: warmup for the first 3% of the time, then cosine down to 10%. a slow mac sees fewer tokens than a fast one, but both finish on time with the schedule fully decayed.

## presets

| preset | params | layers × width | context | time | data |
|---|---|---|---|---|---|
| `smoke` | 0.7M | 2 × 128 | 128 | 3 min | 2M tokens |
| `hour` | 8.0M | 8 × 256 | 512 | 1 hour | 60M tokens |
| `night` | 39.6M | 12 × 512 | 1024 | 8 hours | 400M tokens |

all three fit in 16gb. `--minutes 30` changes the time, and 80% of it goes to pretraining, 20% to chat.

the first run of each preset spends a few minutes downloading and tokenizing before training starts. after that it reuses `data/<preset>`.

## results

| mac | preset | tokens seen | val loss (pretrain) | val loss (chat) | speed start → end |
|---|---|---|---|---|---|
| macbook air m5, 16gb | `hour` | | | | |

every run writes `runs/<preset>/summary.json`. if you run it on a different mac, open an issue with yours and i'll add it.

## your own data

```bash
python train.py --preset hour --text ~/notes
python train.py --preset hour --text book.txt --chat my_chats.jsonl
```

`--text` takes a file or a folder of .txt/.md files. with only that you get a model that continues text in the style of your files (`python chat.py runs/hour --base`). add `--chat` and it learns to reply too. one conversation per line:

```json
{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey!"}]}
```

## stuff to know

macbook airs have no fan, so they slow down once they heat up. the trainer reads macos's thermal state and prints your speed against the first couple of minutes (`-18% vs start`). if it ever hits critical it saves and cools off for 30 seconds. `plot.py` marks the hot parts in red.

it makes things up. 8M params is enough to learn english, not facts.

sampling is cold on purpose (temperature 0.6, min-p 0.1). small models need colder sampling than big ones, which is what my current paper is about. try `--temp 0.3` and `--temp 1.0` to see the difference.

it keeps the mac awake while it trains, but closing the lid still puts it to sleep, so leave it open and plugged in.

ctrl-c saves and prints the exact command to carry on, clock included. it also checkpoints every 10 minutes. `--restart` throws a run away and starts over.

`python plot.py runs/hour` saves loss and speed over time to `runs/hour/plot.png`.

## files

| file | |
|---|---|
| `train.py` | training loop, clock-based schedule, logging, checkpoints |
| `model.py` | the transformer: rmsnorm, rope, grouped-query attention, swiglu, tied embeddings, kv cache |
| `tokenizer.py` | byte-level bpe |
| `prepare.py` | downloads, tokenizes and packs the data |
| `chat.py` | talk to it |
| `config.py` | presets |
| `mac.py` | chip, memory, fanless check, thermal state |
| `plot.py` | loss and speed charts |

## credits

[nanogpt](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat) and [minimind](https://github.com/jingyaogong/minimind) for the idea. tinystories (eldan & li) and hugging face's smollm datasets for the data. apple's mlx for making this possible on a laptop.

mit license
