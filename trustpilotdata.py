from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


def clean_str(s):
    return s.replace('  ', '').replace('\n', '')


def clean_date_time(d):
    d = d[19:38].replace('T', ' ')
    return datetime.strptime(d, '%Y-%m-%d %H:%M:%S')


def get_data(url, page):
    """
    params:
    url: string url of trustpilot
    page: # of pages to import
    """


    user_name = []
    number_of_comments = []
    comment_score = []
    comment_date_time = []
    comment_title = []
    comment_str = []

    for p in range(page):

        url_changer = url + str(p + 1)
        data = urlopen(Request(url_changer))
        soup = BeautifulSoup(data, 'html.parser')

        user_name_finder = soup.find_all('div', class_='consumer-information__name')
        number_of_comments_finder = soup.find_all('div', class_='consumer-information__review-count')
        comment_score_finder = soup.find_all(name='script', attrs={'type':'application/json', 'data-initial-state':'review-info'})
        comment_date_finder = soup.find_all('script', attrs={'type': 'application/json', 'data-initial-state': 'review-dates'})
        comment_title_finder = soup.find_all('h2', class_='review-content__title')
        comment_str_finder = soup.find_all('p', class_='review-content__text')

        for i in range(len(user_name_finder)):

            name = clean_str(user_name_finder[i].text)
            number = int(number_of_comments_finder[i].span.text[:-7])
            score = int(comment_score_finder[i].text[-3])
            tempdate = clean_date_time(comment_date_finder[i].text)
            title = comment_title_finder[i].a.text
            comment = clean_str(comment_str_finder[i].text)

            user_name.append(name)
            number_of_comments.append(number)
            comment_score.append(score)
            comment_date_time.append(tempdate)
            comment_title.append(title)
            comment_str.append(comment)

    df = pd.DataFrame(data=[comment_date_time, number_of_comments, comment_score, comment_title, comment_str],
                      columns=user_name)
    df = df.T
    df.columns = ['comment_date_time', 'comment_count', 'comment_score', 'title', 'comment']

    return df



if __name__ == '__main__':
    url = 'https://www.trustpilot.com/review/www.hsbc.co.uk?page='
    df = get_data(url, 80)
    

