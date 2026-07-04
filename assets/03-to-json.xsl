<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:alto="http://www.loc.gov/standards/alto/ns-v4#"
    exclude-result-prefixes="xs alto"
    version="1.0">

    <!-- Like 01-simplify.xsl, but keeps the geometry (page dims, region and
         line boxes) needed by worker-convert-json.py. All coordinate
         attributes are always emitted (possibly empty). -->

    <!-- Output formatting -->
    <xsl:output indent="yes" method="xml" encoding="UTF-8"/>

    <!-- Define a key for faster lookup -->
    <xsl:key name="label" match="alto:OtherTag" use="@ID" />

    <!-- Template to match the root 'alto' element -->
    <xsl:template match="/alto:alto">
        <doc>
            <xsl:attribute name="width">
                <xsl:value-of select="alto:Layout/alto:Page/@WIDTH"/>
            </xsl:attribute>
            <xsl:attribute name="height">
                <xsl:value-of select="alto:Layout/alto:Page/@HEIGHT"/>
            </xsl:attribute>
            <!-- Apply templates to all TextBlock elements -->
            <xsl:apply-templates select="//alto:TextBlock"/>
        </doc>
    </xsl:template>

    <!-- Shared geometry attributes for regions and lines -->
    <xsl:template name="box">
        <xsl:attribute name="x">
            <xsl:value-of select="@HPOS"/>
        </xsl:attribute>
        <xsl:attribute name="y">
            <xsl:value-of select="@VPOS"/>
        </xsl:attribute>
        <xsl:attribute name="width">
            <xsl:value-of select="@WIDTH"/>
        </xsl:attribute>
        <xsl:attribute name="height">
            <xsl:value-of select="@HEIGHT"/>
        </xsl:attribute>
    </xsl:template>

    <!-- Template to match TextBlock -->
    <xsl:template match="alto:TextBlock">
        <region>
            <!-- Using the key to lookup 'OtherTag' by ID -->
            <xsl:attribute name="type">
                <xsl:value-of select="key('label', @TAGREFS)/@LABEL"/>
            </xsl:attribute>
            <xsl:call-template name="box"/>
            <!-- Apply templates to nested TextLine elements -->
            <xsl:apply-templates select=".//alto:TextLine"/>
        </region>
    </xsl:template>

    <!-- Template to match TextLine -->
    <xsl:template match="alto:TextLine">
        <line>
            <!-- Again, use the key to lookup 'OtherTag' by ID -->
            <xsl:attribute name="type">
                <xsl:value-of select="key('label', @TAGREFS)/@LABEL"/>
            </xsl:attribute>
            <xsl:call-template name="box"/>
            <!-- Apply templates to String and SP elements -->
            <xsl:apply-templates select="alto:String|alto:SP" />
        </line>
    </xsl:template>

    <!-- Template to match String element -->
    <xsl:template match="alto:String">
        <xsl:value-of select="@CONTENT"/>
    </xsl:template>

    <!-- Template to match SP (space) element -->
    <xsl:template match="alto:SP">
        <xsl:text> </xsl:text>
    </xsl:template>

</xsl:stylesheet>
