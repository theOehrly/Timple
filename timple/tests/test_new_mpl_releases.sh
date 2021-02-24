# This script can be called periodically to test against future versions of
# matplotlib.
# If the script finds a new matplotlib release, it will reinstall matplotlib
# from the new release sources. After that, all tests will be run.

# get version number and download link for latest matplotlib release sources
CURRENT_VERSION=$(curl -s https://api.github.com/repos/matplotlib/matplotlib/releases/latest \
| grep "tag_name" \
| awk '{print substr($2, 2, length($2)-3)}')

LOCATION=$"https://github.com/matplotlib/matplotlib/archive/$CURRENT_VERSION.tar.gz"

if test -f "last_mpl_version.txt"
then
  LAST_VERSION=$(<last_mpl_version.txt)
else
  LAST_VERSION=$"unknown"
fi

if [ "$CURRENT_VERSION" == "$LAST_VERSION" ]
then
  # no new matplotlib release to test against
  echo "No new release of matplotlib to test against."
  exit 0
fi

# install newest mpl release from sources and run tests
echo "Found a new release of matplotlib!"
echo "New version is $CURRENT_VERSION"

# uninstall currently installed source version
echo "Uninstalling old version"
python3 -m pip uninstall -y matplotlib
rm -rf ./matplotlib

# download new version
echo "Downloading new version"
curl -L -o matplotlib.tar.gz "$LOCATION"
tar -xf matplotlib.tar.gz
rm matplotlib.tar.gz

# go into matplotlib directory, create setup.cfg to include test baseline images
# and install from sources
cd matplotlib* || exit 1

echo -e "[packages] \ntests = True" > setup.cfg

python3 -m pip install .

cd ..
echo "$CURRENT_VERSION" > last_mpl_version.txt  # remember version

# run tests
echo "New version installed, starting tests"
python3 -m pip install -r requirements.txt  # ensure package requirements
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -v timple/tests
