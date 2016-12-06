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
git add . & git commit -m 'update' & git push


##Markdown tips

###1. Not Realtime Markdown Preview in Subline 

1) Install [Package Contral](https://packagecontrol.io/installation) in Subline

2) Install [Python-Markdown](https://github.com/waylan/Python-Markdown/blob/master/INSTALL.md) as the local markdown complier.
```
pip install markdown
```
if no root, use
```
pip install markdown --user
```

3) Install [Markdown Preview](https://github.com/revolunet/sublimetext-markdown-preview) in subline through **Package Contral**. Use <kbd>cmd</kbd>+<kbd>shift</kbd>+<kbd>P</kbd> then search `Markdown Preview`.

  use

  **Simple Usage :**
  
  a) To preview :
  
  - use <kbd>cmd</kbd>+<kbd>shift</kbd>+<kbd>P</kbd> then `Markdown Preview` to show the commands.
  - or bind some key in your user key binding, using a line like this one:
     `{ "keys": ["alt+m"], "command": "markdown_preview", "args": {"target": "browser", "parser":"markdown"} },` for a specific parser and target or `{ "keys": ["alt+m"], "command": "markdown_preview_select", "args": {"target": "browser"} },` to bring up the quick panel to select enabled parsers for a given target.

  b) To build :
  
  - Just use <kbd>ctrl</kbd>+<kbd>B</kbd> (Windows/Linux) or <kbd>cmd</kbd>+<kbd>B</kbd> (Mac) to build current file.
  
  c) To config :
  
  Using Sublime Text menu: `Preferences`->`Package Settings`->`Markdown Preview`
  
  - `Settings - User` is where you change your settings for Markdown Preview.
  - `Settings - Default` is a good reference with detailed descriptions for each setting.

###2. Realtime Markdown Preview in Subline see [link](https://github.com/yyjhao/markmon)
