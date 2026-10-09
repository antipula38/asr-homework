# Based on seminar materials

# Don't forget to support cases when target_text == ''

def levenshtein_dist(a, b) -> int:
    n = len(a)
    m = len(b)
    dist = [[0 for j in range(n + 1)] for i in range(m + 1)]

    for j in range(n + 1):
        dist[0][j] = j
    
    for i in range(m + 1):
            dist[i][0] = i
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
             dist[i][j] = min(
                  dist[i - 1][j] + 1,
                  dist[i][j - 1] + 1,
                  dist[i - 1][j - 1] + int(a[j - 1] != b[i - 1])
             )
    return dist[m][n]



def calc_cer(target_text, predicted_text):
    if target_text == "":
        if predicted_text == "":
            return {'dist': 0, 'length': len(target_text)}
        else:
             return {'dist': len(predicted_text), 'length': len(target_text)}

    dist = levenshtein_dist(target_text, predicted_text)

    return {'dist': dist, 'length': len(target_text)}




def calc_wer(target_text, predicted_text):
    if len(target_text.split()) == 0:
        len_predicted_words = len(predicted_text.split())
        if len_predicted_words == 0:
            return {'dist': 0, 'length': len(target_text.split())}
        else:
            return {'dist': len_predicted_words, 'length': len(target_text.split())}
        
    dist = levenshtein_dist(target_text.split(), predicted_text.split())
    return {'dist': dist, 'length': len(target_text.split())}
