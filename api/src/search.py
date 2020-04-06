from serpwow.google_search_results import GoogleSearchResults
import subprocess
import json
import os

# create the serpwow object, passing in our API key
api_key = "335A87384FFD4031B11ED9015CA706F4"
serpwow = GoogleSearchResults(api_key)


def search_portraits():
    requests = {
        "need": {
            "markers": [
                "markers portrait",
                "портрет маркеры"
            ]
        },
        "all": {
            "watercolor": [
                "watercolor portrait",
                "портрет акварель"
            ],
            "pencil": [
                "pencil portrait",
                "портрет карандаш"
            ],
            "coal": [
                "coal portrait",
                "портрет уголь"
            ],
            "sanguina": [
                "sanguina portrait",
                "портрет сангина"
            ],
            "sepia": [
                "sepia portrait",
                "портрет сепия"
            ],
            "oil": [
                "oil portrait",
                "портрет масло"
            ],
            "gouache": [
                "gouache portrait",
                "портрет гуашь"
            ],
            "pen": [
                "pen portrait",
                "портрет ручка"
            ],
            "markers": [
                "markers portrait",
                "портрет маркеры"
            ],
            "acrylic": [
                "acrylic portrait",
                "портрет акрил"
            ],
            "tempera": [
                "tempera portrait",
                "портрет темпера"
            ]
        }
    }
    path_to_images = "api/images/data"

    for style in requests["need"].keys():
        if not os.path.exists(path_to_images + style):
            os.mkdir(path_to_images + style)
        path_to_portraits = path_to_images + style + "/portrait_"
        cnt = 1

        for request in requests["need"][style]:
            params = {
                "q": request,
                "search_type": "images"
            }
            result = serpwow.get_json(params)
            # print(json.dumps(result, indent=2, sort_keys=True))
            print('Find images by request "%s":' % request, len(result["image_results"]))

            for image in result["image_results"]:
                try:
                    image_format = '.' + image["image"].split(".")[-1].split('?')[0]
                    subprocess.call(["wget", image["image"], "-O", path_to_portraits + str(cnt) + image_format], timeout=60)
                    cnt += 1
                except Exception as e:
                    print(e)


if __name__ == '__main__':
    search_portraits()
