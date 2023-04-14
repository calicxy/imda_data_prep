import re

def cleanup_string(line):

    words_to_remove = ['(ppo)','(ppc)', '(ppb)', '(ppl)', '<s/>','<c/>','<q/>', '<fil/>', '<sta/>', '<nps/>', '<spk/>', '<non/>', '<unk>', '<s>', '<z>', '<nen>']

    formatted_line = re.sub(r'\s+', ' ', line).strip().lower()

    #detect all word that matches words in the words_to_remove list
    for word in words_to_remove:
        if re.search(word,formatted_line):
            # formatted_line = re.sub(word,'', formatted_line)
            formatted_line = formatted_line.replace(word,'')
            formatted_line = re.sub(r'\s+', ' ', formatted_line).strip().lower()
            # print("*** removed words: " + formatted_line)

    #detect '\[(.*?)\].' e.g. 'Okay [ah], why did I gamble?'
    #remove [ ] and keep text within
    if re.search('\[(.*?)\]', formatted_line):
        formatted_line = re.sub('\[(.*?)\]', r'\1', formatted_line).strip()
        #print("***: " + formatted_line)

    #detect '\((.*?)\).' e.g. 'Okay (um), why did I gamble?'
    #remove ( ) and keep text within
    if re.search('\((.*?)\)', formatted_line):
        formatted_line = re.sub('\((.*?)\)', r'\1', formatted_line).strip()
        # print("***: " + formatted_line)

    #detect '\'(.*?)\'' e.g. 'not 'hot' per se'
    #remove ' ' and keep text within
    if re.search('\'(.*?)\'', formatted_line):
        formatted_line = re.sub('\'(.*?)\'', r'\1', formatted_line).strip()
        #print("***: " + formatted_line)

    #remove punctation '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    punctuation = '''!–;"\,./?@#$%^&*~''' 
    punctuation_list = str.maketrans("","",punctuation)
    formatted_line = re.sub(r'-', ' ', formatted_line)
    formatted_line = re.sub(r'_', ' ', formatted_line)
    formatted_line = formatted_line.translate(punctuation_list)
    formatted_line = re.sub(r'\s+', ' ', formatted_line).strip().lower()
    #print("***: " + formatted_line)

    return formatted_line


if __name__ == "__main__":
    line = "<fil> you can go first (ppc) you guys are going to <fil> stand here <fil>"
    print(cleanup_string(line))