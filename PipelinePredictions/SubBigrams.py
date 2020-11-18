import re
# THIS CODE IS A FAST IMPLEMENTATION OF BIGRAMS MATCHING
#https://stackoverflow.com/questions/42742810/speed-up-millions-of-regex-replacements-in-python-3/42747503#42747503

#for testing
# sentences = ["I'm eric. Welcome here!", "Another boring sentence.",
#                  "GiraffeElephantBoat", "sfgsdg sdwerha aswertwe", 'dkdkdk sss kdkdk animal king sdf sff social sec',
#                  'dkdkdk sss kdkdk zz animal king sdf sff social sec']
#
# bigramset = set(['social sec', 'animal king', 'sss'])

def substituteBigwithMono(sentence, bigramset):
    if len(sentence.split(' ')) < 2:
        print('testo troppo corto, warning')
        return sentence
    #this regex matches couples (but not overlapping) so i need two passes of regex sub
    word_pattern = re.compile('(\w+)\s+(\w+)')

    def delete_banned_words(matchobj):
        word = matchobj.group(0)
        if word.lower() in bigramset:
            return ''.join(word.split(' '))
        else:
            return word



    #first pass matches pairs starting from token 0
    sentence = word_pattern.sub(delete_banned_words, sentence)

    #second pass matches pairs starting from token 1
    firstword, sent = sentence.split(' ', 1)
    sent = word_pattern.sub(delete_banned_words, sent)
    sentence = firstword + ' ' + sent
    return sentence