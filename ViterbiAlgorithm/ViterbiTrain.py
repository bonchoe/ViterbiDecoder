def train_initial_distribution(tag_probs, states, emit_p, tagged_tokens):
    for token, tag in tagged_tokens:

        if tag not in tag_probs:
            tag_probs[tag] = dict()
            tag_probs[tag]['count_tag'] = dict()
            tag_probs[tag]['occurence'] = 1
            if tag not in states:
                states.append(tag)
        else:
            tag_probs[tag]['occurence'] += 1
            
        if token not in tag_probs[tag]['count_tag']:
            tag_probs[tag]['count_tag'][token] = 1
        else:
            tag_probs[tag]['count_tag'][token] += 1
            
    for tag in tag_probs:
        tag_probs[tag]['probs'] = dict()
        for token in tag_probs[tag]['count_tag']:
            tag_probs[tag]['probs'][token] = float(tag_probs[tag]['count_tag'][token]) / float(tag_probs[tag]['occurence'])
            if tag not in emit_p:
                emit_p[tag] = dict()
            emit_p[tag][token] = tag_probs[tag]['probs'][token]
        
def construct_transition_probabilities_per_tag(start_p, transition_probs, trans_p, tagged_tokens):
    """
    input:
    -------
    tagged_tokens: list of tagged tokens in form like:
            [('The', 'AT'),
             ('Fulton', 'NP-TL'),
             ('County', 'NN-TL'),
             ('Grand', 'JJ-TL'),
             ('Jury', 'NN-TL'),
             ('said', 'VBD'),
             ('Friday', 'NR'),
             ('an', 'AT'),
             ('investigation', 'NN'),
             ('of', 'IN')]
             
    This function will generate the emission matrix / probabilities for each tag
        
    """
    # transition_probs = dict()
    for i in range(len(tagged_tokens)):
        current_tag = tagged_tokens[i][1]
        if current_tag not in transition_probs:
            transition_probs[current_tag] = dict()
            transition_probs[current_tag]['occurence'] = 0
            transition_probs[current_tag]['count_transition'] = dict()
            
        if current_tag not in start_p:
            start_p[current_tag] = 1
        else:
            start_p[current_tag] += 1
        
        # evaluate previous tag
        if i > 0:
            previous_tag = tagged_tokens[i-1][1]
            pt = tagged_tokens[i-1][0]
            transition_probs[previous_tag]['occurence'] += 1
            
            # special case for <start> tag
            if pt == '.':
                if '<start>' not in transition_probs:
                    transition_probs['<start>'] = dict()
                    transition_probs['<start>']['occurence'] = 0
                    transition_probs['<start>']['count_transition'] = dict()
                if current_tag not in transition_probs['<start>']['count_transition']:
                    transition_probs['<start>']['count_transition'][current_tag] = 0
                    
                transition_probs['<start>']['count_transition'][current_tag] += 1
                transition_probs['<start>']['occurence'] += 1
                    
            
            #init
            if current_tag not in transition_probs[previous_tag]['count_transition']:
                transition_probs[previous_tag]['count_transition'][current_tag] = 0
                
            transition_probs[previous_tag]['count_transition'][current_tag] += 1
    for tag in transition_probs:
        transition_probs[tag]['probs'] = dict()
        for transit_tag in transition_probs[tag]['count_transition']:
            transition_probs[tag]['probs'][transit_tag] = \
                float(transition_probs[tag]['count_transition'][transit_tag]) / float(transition_probs[tag]['occurence'])
            if tag not in trans_p:
                trans_p[tag] = dict()
            trans_p[tag][transit_tag] = transition_probs[tag]['probs'][transit_tag]
    
    total = sum(start_p.values())
    start_p = {key: value / total for key, value in start_p.items()}
    return start_p

def word_available(word, emit_p):
    result = False
    for st in emit_p.keys():
        result = result | (word in emit_p[st])
    return result