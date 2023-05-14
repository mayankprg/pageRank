import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(sum(ranks.values()))
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(sum(ranks.values()))

    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    model = dict()

    if len(corpus[page]) == 0:
        for key in corpus:
            model[key] = round(1 / len(corpus), 4)
        return model

    for link in corpus[page]:
        model[link] = round(damping_factor / len(corpus[page]), 4)

    for link in corpus:
        if link in model:
            model[link] = round(
                model[link] + ((1 - damping_factor) / len(corpus)), 4)
        else:
            model[link] = round((1 - damping_factor) / len(corpus), 4)

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # initialize sample dictionary
    data = dict()
    for key in corpus:
        data[key] = 0

    # get first sample randomly
    sample = [random.choice(list(corpus))]
    data[sample[0]] += 1

    # get next sample from previous sample's transtion_model
    for _ in range(n-1):
        modal = transition_model(
            corpus=corpus, page=sample[0], damping_factor=damping_factor)
        sample = random.choices(list(modal.keys()), modal.values(), k=1)
        data[sample[0]] += 1

    rank = dict()
    for key in data:
        rank[key] = round(data[key] / n, 4)

    return rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # N
    total_pages_corpus = len(corpus)

    # every page's pageRank set ot 1/N
    rank = dict()
    for key in corpus.keys():
        rank[key] = round(1 / total_pages_corpus, 4)

    while (True):

        delta_ranks = copy.deepcopy(rank)

        for page in corpus.keys():

            if len(corpus[page]) == 0:
                current_rank = round(((1 - damping_factor) / total_pages_corpus) *
                                     (sum(delta_ranks.values()) / len(corpus)), 4)
            else:
                links = get_page_link(page, corpus)
                links_page_ranks = 0
                for link in links:
                    links_page_ranks = links_page_ranks + (delta_ranks[link] /
                                                           num_links(link, corpus))
                current_rank = round(
                    ((1 - damping_factor) / total_pages_corpus) + (damping_factor * (links_page_ranks)), 10)
                rank[page] = current_rank

        count = 0
        for page in rank:
            if round(delta_ranks[page] - rank[page], 4) < 0.001 and round(rank[page] - delta_ranks[page], 4) < 0.001:
                count += 1
        if count == len(corpus):
            break
    return rank


def get_page_link(page, corpus):
    links = list()
    for link in corpus.keys():
        if page in corpus[link]:
            links.append(link)
    return links


def num_links(page, corpus):
    return len(corpus[page])


if __name__ == "__main__":
    main()
