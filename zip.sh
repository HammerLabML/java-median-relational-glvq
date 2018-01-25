set -e
mkdir median_relational_glvq
cp -r gpl-3.0.md gpl_license_header_mrglvq.txt javadoc pom.xml README.md mrglvq-0.1.0.jar median_relational_glvq/.
zip -r mrglvq-0.1.0.zip median_relational_glvq/*
rm -rf median_relational_glvq
