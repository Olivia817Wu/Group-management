from django.db import models
from django.db.models import Avg


# 分类信息表
class Genre(models.Model):
    name = models.CharField(max_length=100, verbose_name="type")

    class Meta:
        db_table = 'Genre'
        verbose_name = 'Movie type'
        verbose_name_plural = 'Movie type'

    def __str__(self):
        return self.name


# 电影信息表
class Movie(models.Model):
    name = models.CharField(max_length=256, verbose_name="Film title")
    imdb_id = models.IntegerField(verbose_name="imdb_id")
    time = models.CharField(max_length=256, blank=True, verbose_name="Time")
    genre = models.ManyToManyField(Genre, verbose_name="Type")
    release_time = models.CharField(max_length=256, blank=True, verbose_name="release_time")
    intro = models.TextField(blank=True, verbose_name="intro")
    director = models.CharField(max_length=256, blank=True, verbose_name="director")
    writers = models.CharField(max_length=256, blank=True, verbose_name="writers")
    actors = models.CharField(max_length=512, blank=True, verbose_name="cctors")
    # 电影和电影之间的相似度,A和B的相似度与B和A的相似度是一致的，所以symmetrical设置为True
    movie_similarity = models.ManyToManyField("self", through="Movie_similarity", symmetrical=False,
                                              verbose_name="Similar film")

    class Meta:
        db_table = 'Movie'
        verbose_name = 'Movie information'
        verbose_name_plural = 'Movie information'

    def __str__(self):
        return self.name

    # 获取平均分的方法
    def get_score(self):
        result_dct = self.movie_rating_set.aggregate(Avg('score'))  # 格式 {'score__avg': 3.125}
        try:
            result = round(result_dct['score__avg'], 1)  # 只保留一位小数
        except TypeError:
            return 0
        else:
            return result

    # 获取用户的打分情况
    def get_user_score(self, user):
        return self.movie_rating_set.filter(user=user).values('score')

    # 整数平均分
    def get_score_int_range(self):
        return range(int(self.get_score()))

    # 获取分类列表
    def get_genre(self):
        genre_dct = self.genre.all().values('name')
        genre_lst = []
        for dct in genre_dct.values():
            genre_lst.append(dct['name'])
        return genre_lst

    # 获取电影的相识度
    def get_similarity(self, k=5):
        # 默认获取5部最相似的电影的id
        similarity_movies = self.movie_similarity.all()[:k]
        return similarity_movies


# 电影相似度
class Movie_similarity(models.Model):
    movie_source = models.ForeignKey(Movie, related_name='movie_source', on_delete=models.CASCADE, verbose_name="Source film")
    movie_target = models.ForeignKey(Movie, related_name='movie_target', on_delete=models.CASCADE, verbose_name="Target film")
    similarity = models.FloatField(verbose_name="similarity")

    class Meta:
        # 按照相似度降序排序
        verbose_name = 'Movie_similarity'
        verbose_name_plural = 'Movie_similarity'


# 用户信息表
class User(models.Model):
    name = models.CharField(max_length=128, unique=True, verbose_name="User name")
    password = models.CharField(max_length=256, verbose_name="password")
    email = models.EmailField(unique=True, verbose_name="email")
    rating_movies = models.ManyToManyField(Movie, through="Movie_rating")

    def __str__(self):
        return "<USER:( name: {:},password: {:},email: {:} )>".format(self.name, self.password, self.email)

    class Meta:
        db_table = 'User'
        verbose_name = 'User name'
        verbose_name_plural = 'User name'


# 电影评分信息表
class Movie_rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, unique=False, verbose_name="User")
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, unique=False, verbose_name="Movie")
    score = models.FloatField(verbose_name="Score")
    comment = models.TextField(blank=True, verbose_name="Comment")

    class Meta:
        db_table = 'Movie_rating'
        verbose_name = 'Movie rating information'
        verbose_name_plural = 'Movie rating information'


# 最热门的一百部电影
class Movie_hot(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, verbose_name="Movie name")
    rating_number = models.IntegerField(verbose_name="Number of graders")

    class Meta:
        db_table = 'Movie_hot'
        verbose_name = 'hottest movie'
        verbose_name_plural = 'hottest movie'

# python manage.py makemigrations
# python manage.py migrate
