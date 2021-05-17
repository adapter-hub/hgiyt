import glob
import os
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# list of cjk codepoint ranges
# tuples indicate the bottom and top of the range, inclusive
cjk_ranges = [
    (0x4E00, 0x62FF),
    (0x6300, 0x77FF),
    (0x7800, 0x8CFF),
    (0x8D00, 0x9FCC),
    (0x3400, 0x4DB5),
    (0x20000, 0x215FF),
    (0x21600, 0x230FF),
    (0x23100, 0x245FF),
    (0x24600, 0x260FF),
    (0x26100, 0x275FF),
    (0x27600, 0x290FF),
    (0x29100, 0x2A6DF),
    (0x2A700, 0x2B734),
    (0x2B740, 0x2B81D),
    (0x2B820, 0x2CEAF),
    (0x2CEB0, 0x2EBEF),
    (0x2F800, 0x2FA1F),
]


def is_cjk(char):
    char = ord(char)
    for bottom, top in cjk_ranges:
        if char >= bottom and char <= top:
            return True
    return False


def analyze_UD_file(file_names, tokenizer):

    used_vocab = defaultdict(int)

    # how many words have been pre-tokenized into conll format
    word_counter = 0

    # the number of tokens a single word has been tokenized into
    split_lengths = []

    # list of first and continuation words used in the dataset
    first_words = []
    continuation = []

    # number of tokens that include an unk
    unk_inside = 0

    # whether a word needed to be split or its full representation was in the vocabulary
    full_length = 0
    continuation_word = 0

    # list of pairs of [actual_s_length, sentence_length]
    s_length_list = []

    for file_name in file_names:
        print(file_name)
        with open(file_name, "r", encoding="utf-8") as f:

            # sentence length with tokenizer
            sentence_length = 0

            # sentence length in conll
            actual_s_length = 0

            cjk_count = 0
            word_count_file = 0
            cjks = []

            for line in f:
                # if sentence split (new line)
                if len(line) < 2:
                    # if previous sentence is finished, add the last one's information and reset
                    if sentence_length > 0:
                        s_length_list.append([actual_s_length, sentence_length])
                        sentence_length = 0
                        actual_s_length = 0
                    continue

                # if meta data line, continue
                if line.startswith("#"):
                    continue

                elems = line.split()

                # some datasets have multiple tokenization versions, e.g. for compound words. We skip those
                if "-" in elems[0]:
                    continue

                # conll has the actual token in position 1
                token = elems[1]

                if any(map(is_cjk, token)):
                    cjk_count += 1
                    cjks.append("".join(filter(is_cjk, token)))
                    # continue

                # tokenize the word with the passed tokenizer
                ids = tokenizer.encode(token)[1:-1]
                length = len(ids)
                if length == 0:
                    continue
                # if ids[0] == 6:
                #    ids = ids[1:]

                # length of actual splits of a true token
                splits = tokenizer.convert_ids_to_tokens(ids)

                for sub_word in splits:
                    used_vocab[sub_word] += 1

                length = len(splits)

                if tokenizer.unk_token_id in ids:
                    unk_inside += 1

                # whether a word needed to be split or its full representation was in the vocabulary
                if length == 1:
                    full_length += 1
                else:
                    continuation_word += 1

                # count up number of words in dataset
                word_counter += 1
                word_count_file += 1
                # count up the current conll sentence length
                actual_s_length += 1

                # add number of tokens the tokenizer has actually created
                sentence_length += length

                # add the length of the tokenizer to be able to average over how many tokens single words have been tok.
                split_lengths.append(length)

                # add tokens to list of first words and continuation words
                try:
                    first_words += [splits[0]]
                    continuation += splits[1:]
                except:
                    print(f"token: <{token}>, ids: <{ids}>, splits: <{splits}>")
        print(f"CJK Count: {cjk_count}")
        print(f"Word Count: {word_count_file}")

    return (
        used_vocab,
        word_counter,
        split_lengths,
        s_length_list,
        first_words,
        continuation,
        full_length,
        continuation_word,
        unk_inside,
    )


def get_meta_data_for_languages(language_ud_dict, tokenizers):
    for language, d in language_ud_dict.items():
        tokenizer = tokenizers[language] if len(tokenizers) > 1 else tokenizers["mBERT"]
        (
            used_vocab,
            word_counter,
            split_lengths,
            s_length_list,
            first_words,
            continuation,
            full_length,
            continuation_word,
            unk_inside,
        ) = analyze_UD_file(d["files"], tokenizer)
        d["used_vocab"] = used_vocab
        d["word_counter"] = word_counter
        d["split_lengths"] = split_lengths
        d["s_length_list"] = s_length_list
        d["first_words"] = first_words
        d["continuation"] = continuation
        d["full_length"] = full_length
        d["continuation_word"] = continuation_word
        d["unk_inside"] = unk_inside


def plot_set_continuation(language_ud_dict):

    languages = []
    values = []
    for k, v in language_ud_dict.items():
        languages.append(k)
        values.append(len(set(v["continuation"])) / (len(set(v["continuation"])) + len(set(v["first_words"]))))
    d = {"languages": languages, "set continuation proportion": values}
    df = pd.DataFrame(data=d).sort_values(ascending=True, by="set continuation proportion")


def plot_continuation(language_ud_dict):
    sns.set(style="whitegrid")

    width = 512.14963

    colors = ["#B90F22", "#00689D", "#008877", "#951169", "#D7AC00", "#B1BD00", "#CC4C03", "#611C73", "#7FAB16"]
    sns.set_palette(sns.color_palette(colors))
    sns.set_context("notebook")  # use notebook or talk

    fig.set_size_inches(11.7, 8.27)

    languages = []
    values = []
    for k, v in language_ud_dict.items():
        languages.append(k)
        values.append(len(v["continuation"]) / (len(v["continuation"]) + len(v["first_words"])))
    d = {"Language": languages, "Continuation proportion": values}
    df = pd.DataFrame(data=d).sort_values(ascending=True, by="Language")
    ax = sns.barplot(x="Language", y="Continuation proportion", data=df, ax=ax)


def plot_proportion_continuation(language_ud_dicts):
    sns.set(style="whitegrid")

    sns.set(
        rc={
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.grid": False,
            "legend.fontsize": 10,
            "ytick.left": True,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "pgf.texsystem": "lualatex",
            "text.latex.preamble": r"\usepackage{xcolor}",
            "text.usetex": True,
        },
        style="whitegrid",
    )

    c = (185 / 255, 15 / 255, 34 / 255)

    colors = ["indianred", "skyblue", "dodgerblue", "royalblue", "navy"]
    sns.set_palette(sns.color_palette(colors))
    sns.set_context("notebook")

    titles = ["Mono", "mBERT"]
    for i, language_ud_dict in enumerate(language_ud_dicts):

        languages = []
        values = []
        for k, v in language_ud_dict.items():
            languages.append(r"\textsc{%s}" % k)
            values.append(v["continuation_word"] / (v["continuation_word"] + v["full_length"]))
        d = {"languages": languages, "proportion continuation": values}
        df = pd.DataFrame(data=d).sort_values(ascending=True, by="proportion continuation")

    d = {"Language": [], "Proportion of continued words": [], "Model": []}
    for i, language_ud_dict in enumerate(language_ud_dicts):

        languages = []
        values = []
        for k, v in language_ud_dict.items():
            languages.append(r"\textsc{%s}" % k)
            values.append(v["continuation_word"] / (v["continuation_word"] + v["full_length"]))
        d["Language"] += languages
        d["Proportion of continued words"] += values
        d["Model"] += [titles[i] for _ in values]
    df = pd.DataFrame(data=d).sort_values(ascending=True, by="Language")

    ax2 = sns.catplot(
        kind="bar",
        x="Language",
        y="Proportion of continued words",
        hue="Model",
        data=df,
        legend=False,
        height=5,
        aspect=2.1,
    )

    plt.legend(title="Model", title_fontsize=30, loc="center right", bbox_to_anchor=(1.3, 0.5), fontsize=28)
    ax2.set_xlabels("")
    ax2.set_ylabels(fontsize=30)
    ax2.set_xticklabels(fontsize=30)
    ax2.set(yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax2.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], fontsize=28)

    ax2.savefig("continuation.pdf", bbox_inches="tight")

    return df


def plot_fertility(language_ud_dicts):
    sns.set(style="whitegrid")

    width = 512.14963
    sns.set(
        rc={
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.grid": False,
            "legend.fontsize": 10,
            "ytick.left": True,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "pgf.texsystem": "lualatex",
            "text.latex.preamble": r"\usepackage{xcolor}",
            "text.usetex": True,
        },
        style="whitegrid",
    )

    colors = ["indianred", "skyblue", "dodgerblue", "royalblue", "navy"]
    sns.set_palette(sns.color_palette(colors))
    sns.set_context("notebook")  # use notebook or talk

    titles = ["Mono", "mBERT"]
    for i, language_ud_dict in enumerate(language_ud_dicts):

        languages = []
        values = []
        for k, v in language_ud_dict.items():
            languages.append(r"\textsc{%s}" % k)
            values.append(np.mean(v["split_lengths"]))
        d = {"languages": languages, "fertility": values}
        df = pd.DataFrame(data=d).sort_values(ascending=True, by="fertility")

    d = {"Language": [], "Fertility": [], "Model": []}
    for i, language_ud_dict in enumerate(language_ud_dicts):

        languages = []
        values = []
        for k, v in language_ud_dict.items():
            languages.append(r"\textsc{%s}" % k)
            values.append(np.mean(v["split_lengths"]))
        d["Language"] += languages
        d["Fertility"] += values
        d["Model"] += [titles[i] for _ in values]
    df = pd.DataFrame(data=d).sort_values(ascending=True, by="Language")

    ax2 = sns.catplot(
        kind="bar", x="Language", y="Fertility", hue="Model", data=df, legend=False, height=5, aspect=2.1
    )

    ax2.set_xlabels("")
    ax2.set_ylabels(fontsize=30)
    ax2.set_xticklabels(fontsize=30)
    ax2.set(yticks=[0.0, 0.5, 1.0, 1.5, 2.0])
    ax2.set_yticklabels([0.0, 0.5, 1.0, 1.5, 2.0], fontsize=28)

    ax2.savefig("fertility.pdf", bbox_inches="tight")

    return df


def plot_proportion_unks(language_ud_dicts):
    sns.set(style="whitegrid")
    sns.set(
        rc={
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.grid": False,
            "legend.fontsize": 10,
            "ytick.left": True,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "pgf.texsystem": "lualatex",
            "text.latex.preamble": r"\usepackage{xcolor}",
            "text.usetex": True,
        },
        style="whitegrid",
    )

    colors = ["indianred", "skyblue", "dodgerblue", "royalblue", "navy"]
    sns.set_palette(sns.color_palette(colors))
    sns.set_context("notebook")  # use notebook or talk

    titles = ["Mono", "mBERT"]
    for i, language_ud_dict in enumerate(language_ud_dicts):

        languages = []
        values = []
        for k, v in language_ud_dict.items():
            languages.append(r"\textsc{%s}" % k)
            values.append(v["unk_inside"] / v["word_counter"])
        d = {"Language": languages, "unk proportion": values}
        df = pd.DataFrame(data=d).sort_values(ascending=True, by="unk proportion")

    d = {"Language": [], "Proportion UNK": [], "Tokenizer": []}
    for i, language_ud_dict in enumerate(language_ud_dicts):

        languages = []
        values = []
        for k, v in language_ud_dict.items():
            languages.append(r"\textsc{%s}" % k)
            values.append(v["unk_inside"] / v["word_counter"])
        d["Language"] += languages
        d["Proportion UNK"] += values
        d["Tokenizer"] += [titles[i] for _ in values]
    df = pd.DataFrame(data=d).sort_values(ascending=True, by="Language")
    ax2 = sns.catplot(
        kind="bar", x="Language", y="Proportion UNK", hue="Tokenizer", data=df, legend=False, height=5, aspect=2.1
    )

    plt.legend(title="Model", title_fontsize=30, loc="center right", bbox_to_anchor=(1.3, 0.5), fontsize=28)
    ax2.set_xlabels("")
    ax2.set_ylabels(fontsize=30)
    ax2.set_xticklabels(fontsize=30)
    ax2.set(yticks=[0.000, 0.004, 0.008, 0.012, 0.016])
    ax2.set_yticklabels([0.000, 0.004, 0.008, 0.012, 0.016], fontsize=28)

    ax2.savefig("unknowns.pdf", bbox_inches="tight")
    return df


def plot_dist_length(language_ud_dict, mBert_dict):
    titles = ["Monolingual", "mBERT"]

    sns.set(style="whitegrid")
    sns.set(
        rc={
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 30,
            "axes.labelsize": 30,
            "axes.grid": False,
            "legend.fontsize": 10,
            "ytick.left": True,
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "pgf.texsystem": "lualatex",
            "text.latex.preamble": r"\usepackage{xcolor}",
            "text.usetex": True,
        },
        style="whitegrid",
    )

    colors = ["indianred", "skyblue", "dodgerblue", "royalblue", "navy"]
    sns.set_palette(sns.color_palette(colors))
    sns.set_context("notebook")  # use notebook or talk

    for k, v in language_ud_dict.items():

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

        ax.set_xlabel("Sentence length [Tokens]", fontsize=30)
        ax.set_ylabel("Proportion", fontsize=30)
        sns.distplot(
            np.array(v["s_length_list"])[:, 0],
            hist=False,
            rug=False,
            ax=ax,
            label="Reference",
            kde_kws={"linewidth": 4},
        )
        sns.distplot(
            np.array(v["s_length_list"])[:, 1],
            hist=False,
            rug=False,
            ax=ax,
            label="Mono",
            kde_kws={"linestyle": "-", "linewidth": 4},
        )

        sns.distplot(
            np.array(mBert_dict[k]["s_length_list"])[:, 1],
            hist=False,
            rug=False,
            ax=ax,
            label="mBERT",
            kde_kws={"linestyle": "--", "linewidth": 4},
        )
        ax.set(xlim=(-10, 200))
        ax.legend(loc="upper right", bbox_to_anchor=(1.03, 1.02), fontsize=28)
        plt.locator_params(axis="y", nbins=6)
        plt.locator_params(axis="x", nbins=6)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)

        ax.figure.savefig(f"{k}_sentence_length.pdf", bbox_inches="tight")
