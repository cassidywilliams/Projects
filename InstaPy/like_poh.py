from instapy import InstaPy
import configparser

config = configparser.ConfigParser()
config.read("C:\Windows\System32\drivers\etc\config.txt")
name = config.get("poh","poh_name")
key = config.get("poh","poh_key")

session = InstaPy(username=name, password=key)
session.login()

session.set_user_interact(amount=3, randomize=True, percentage=100, media='Photo')
session.like_by_feed(amount=342, randomize=True, unfollow=True, interact=True)
session.like_by_locations(['458552132', '1087457797963695', '1044419272262640', '213131048',
    '213110159',
    '397954500',
    '342980',
    '1657921921107183',
    '240013154',
    '226981041',
    '1022845841',
    '217994113',
    '224247299'], amount=100)
session.like_by_tags(['peopleofhamburg', 'europe', 'travel', 'passionpassport', 'profile_vision'
                      'yngkillers', 'HamburgFotografiert', 'seemycity', 'ig_hamburg'
                      '_welovehh', 'hamburgbestof', 'bestofhamburg', 'igershamburg', 'igershh',
                      '365_tage_hamburg', '365_tage_hafenliebe', 'heuteinhamburg',
                      'moinmoin', 'hh_highlights', 'all_shots', 'elbphilharmonie',
                      'weltkulturerbe', 'moin', 'ahoi', 'clouds', 'bluesky',
                      'hamburchmeineperle', 'hamburchbestof', 'deichstrasse', 'hamburch',
                      'hhliebe', 'hh', 'holzbrücke', 'hamburg', 'klassiker', 'awesomehamburg'], amount=100)



session.end()



