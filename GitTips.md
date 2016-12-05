##Git tips

###1. Permanently authenticating with Git repositories, see [stackoverflow](http://stackoverflow.com/questions/6565357/git-push-requires-username-and-password#comment22515527_6565357)

1) For HTTPS, use credential caching
Run following command to enable credential caching.
```
$ git config credential.helper store
$ git push https://github.com/repo.git

Username for 'https://github.com': <USERNAME>
Password for 'https://USERNAME@github.com': <PASSWORD>
```

Use should also specify caching expire,

```
git config --global credential.helper 'cache --timeout 7200'
```

After enabling credential caching, it will be cached for 7200 seconds (2 hour).

2) For SSH, use keys (more **Security**)
A common mistake is cloning using the default (HTTPS) instead of SSH. You can correct this by going to your repository, clicking the ssh button left to the URL field and updating the URL of your origin remote like this:
```
git remote set-url origin git@github.com:username/repo.git
```
Fixed my Permission denied (publickey) using this guide: [help.github.com/articles/generating-ssh-keys](https://help.github.com/articles/checking-for-existing-ssh-keys/).


###2. Common Git Command, see [git-cheat-sheet](http://www.ruanyifeng.com/blog/2015/12/git-cheat-sheet.html)