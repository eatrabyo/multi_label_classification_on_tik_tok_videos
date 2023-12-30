import asyncio
from douyin_tiktok_scraper.scraper import Scraper
import urllib.request
from pprint import pprint

api = Scraper()

async def hybrid_parsing(url: str) -> dict:
    # Hybrid parsing(Douyin/TikTok URL)
    result = await api.hybrid_parsing(url)
    print(f"The hybrid parsing result:\n {result}")
    return result


def download_func(url,root_folder,file_name):
    path = root_folder + file_name

    x = asyncio.run(hybrid_parsing(url= url))
    donwload_url = x['video_data']['nwm_video_url']
    urllib.request.urlretrieve(donwload_url, path)

if __name__ == '__main__':
    url = 'https://www.tiktok.com/@inkzchimmanee/video/7238365029203168518?q=%E0%B9%80%E0%B8%A7%E0%B8%97%E0%B9%80%E0%B8%97%E0%B8%A3%E0%B8%99%E0%B8%99%E0%B8%B4%E0%B9%88%E0%B8%87&t=1703608129020'
    path = 'data/'
    file_name = 'vid_1.mp4'
    download_func(url,path,file_name)