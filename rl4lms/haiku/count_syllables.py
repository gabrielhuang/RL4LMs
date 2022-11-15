
import cmudict  # for syllables
import syllables
import re

whitespace = re.compile(r'[\s,.?!/=();]+')
cmudict_cached = cmudict.dict()


def lookup_word(word_s):
    word = cmudict_cached.get(word_s)
    return word

def count_syllables(word_s, fast=True):
    count = 0
    phones = lookup_word(word_s) # this returns a list of matching phonetic rep's
    if phones and not fast:                   # if the list isn't empty (the word was found)
        phones0 = phones[0]      #     process the first
        count = len([p for p in phones0 if p[-1].isdigit()]) # count the vowels
    else:  # fallback to syllables library
        count = syllables.estimate(word_s) 
        #print(f'counting "{word_s}" -> {count}')
    return count

def count_syllables_in_sentence(sentence, fast=True):
    sentence = sentence.strip('\n\t ').lower()
    words = [x for x in whitespace.split(sentence) if x]
    #words = [x for x in sentence.split(' \n\t\r,.?!/=()') if x]
    # count syllables for each word
    syllables = [count_syllables(w, fast) for w in words]
    return sum(syllables)

def count_syllables_in_haiku(haiku, fast=False):
    '''
    haiku / is separated / with slashes
    '''
    lines = haiku.strip(' \n()[]').split('/')
    return [count_syllables_in_sentence(line, fast) for line in lines]

if __name__ == '__main__':
    sentences = ['hello world!', 'this is blabli', 'said I, without a doubt',]
    for s in sentences:
        print(count_syllables_in_sentence(s, True), s)
    haikus = ['Deep autumn; /	The apple colder/	In the tree.',
    "It's crazy to see / How small my body can get? / And still feel complete",
    "No body gonna / Help you on Twitter until / You can get your shit",
    "First haiku... unintentionally reversed (7 / 5 / 7)",
    "I predicted this / Today's result, but it will / Still be reversed"
    ]
    for h in haikus:
        print(count_syllables_in_haiku(h, False), h)