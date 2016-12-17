import numpy as np


OPEN_LINE = '<l>'
CLOSE_LINE = '</l>'    
OPEN_STANZA = '<s>'
CLOSE_STANZA = '</s>'
TAGS = [OPEN_LINE, CLOSE_LINE, OPEN_STANZA, CLOSE_STANZA]


# Get a list of the tags in order for a given song line 
def get_tags(text):
    # You can't get all the tags if you just split by space. This fixes it.
    text = text.replace('>', '> ').replace('<', ' <')
    split_text = text.split(' ')
    tags = [token for token in split_text if token in TAGS]
    return tags
    

# Returns true if the tags are nested correctly
def tags_balanced(tokens):
    # Determine if the tags are balanced or not (ignore <start>)
    tag_stack = []
    for token in tokens:
        if token not in TAGS:
            continue
        if token == OPEN_LINE or token == OPEN_STANZA:
            tag_stack.append(token)
        elif (tag_stack and
              (token == CLOSE_LINE and tag_stack[-1] == OPEN_LINE) or
              (token == CLOSE_STANZA and tag_stack[-1] == OPEN_STANZA)):
            tag_stack.pop()
        else:
            tag_stack = ['fail']
            break
    # Tags are balance if the stack is empty
    return not tag_stack

    
def tags_balance_score(tokens, open_tags, close_tags):
    score = 0
    for token in tokens:
        if token in open_tags:
            score += 1
        elif token in close_tags:
            score -= 1
    return abs(score)


# Given a list of texts, it prints the average stats
def evaluate_text(texts):
    perfect_balance_list = []
    line_balance_scores = []
    stanza_balance_scores = []
    overall_balance_scores = []
    for text in texts:
        tags = get_tags(text)
        perfect_balance_list.append(tags_balanced(tags))
        line_balance_scores.append(tags_balance_score(tags, [OPEN_LINE], [CLOSE_LINE]))
        stanza_balance_scores.append(tags_balance_score(tags, [OPEN_STANZA], [CLOSE_STANZA]))
        overall_balance_scores.append(tags_balance_score(tags, [OPEN_LINE, OPEN_STANZA],
                                                         [CLOSE_LINE, CLOSE_STANZA]))
    print('Percentage balance: {}'.format(float(sum(perfect_balance_list)) / len(perfect_balance_list)))
    print('Line balancing: mean: {0:0.3f}, std: {1:0.3f}'.format(np.mean(line_balance_scores), np.std(line_balance_scores)))
    print('Stanza balancing: mean: {0:0.3f}, std: {1:0.3f}'.format(np.mean(stanza_balance_scores), np.std(stanza_balance_scores)))
    print('Overall balancing: mean: {0:0.3f}, std: {1:0.3f}'.format(np.mean(overall_balance_scores), np.std(overall_balance_scores)))
