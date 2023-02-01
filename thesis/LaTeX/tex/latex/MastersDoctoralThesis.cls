%----------------------------------------------------------------------%
%    University of Utah LaTeX Updated Thesis Template
%    Christopher Creveling 
%    12-7-2020
%    Department of Mechanical Engineering
%----------------------------------------------------------------------%


%----------------------------------------------------------------------%
%                           Identification
%----------------------------------------------------------------------%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{MastersDoctoralThesis}[12/7/20 MastersDoctoralThesis]

%----------------------------------------------------------------------%
%                           Report options
%----------------------------------------------------------------------%

\DeclareOption{11pt}{
    \PassOptionsToClass{\CurrentOption}{article} % book report
}

%% Execute default options
\ExecuteOptions{11pt}

%% Process given options
\ProcessOptions\relax

%% Load base (force chapters to always start on the right-hand page)
\LoadClass[11pt,letterpaper,twoside]{report} % ,openright

%----------------------------------------------------------------------%
%                 Numbering table of contents
%----------------------------------------------------------------------%

% seting level of numbering (default for "report" is 3). With ''-1'' you 
% have non number also for chapters
\setcounter{secnumdepth}{5}
% if you want all the levels in your table of contents
\setcounter{tocdepth}{5} 

%----------------------------------------------------------------------%
%                           Packages
%----------------------------------------------------------------------%

\RequirePackage[utf8]{inputenc} % Required for inputting international characters
\RequirePackage[T1]{fontenc} % Output font encoding for international characters
\RequirePackage{soul} % Advanced highlighting

%----------------------------------------------------------------------%
%                    Math Related Packages
%----------------------------------------------------------------------%
%\RequirePackage{amsmath}
\RequirePackage{amssymb}
\let\bvec\mathbf % Bold vector definition
\RequirePackage{empheq} % loading `mathtools` % loading `amsmath
%\RequirePackage{cases} % better equation labels (Clash with cleveref)
\RequirePackage{amsbsy} % bold symbols in section header names \texorpdfstring{}{}
\RequirePackage{mathtools}
\DeclarePairedDelimiterX{\norm}[1]{\lVert}{\rVert}{#1} % Vector Norm


%----------------------------------------------------------------------%
%                           Fonts
%----------------------------------------------------------------------%
\RequirePackage{mathpazo} % Use the Palatino font by default


%----------------------------------------------------------------------%
%              Specific University of Utah Spacing
%----------------------------------------------------------------------%

\newlength{\UofUOneInChapterSpace}
\newlength{\UofUHeadSpace}
\newlength{\UofUDoubleSpace}
\newlength{\UofUnegDoubleSpace}
\newlength{\UofUnegSingleSpace}
\makeatletter

\RequirePackage{calc} % Allows addition of different units etc.
% Chapter titles are 2" from the top (Remove the default 40pt space) % University guideline
\setlength{\UofUOneInChapterSpace}{-40pt + 1in} 
\setlength{\UofUHeadSpace}{2ex} % Two blank lines % University guideline
\setlength{\UofUDoubleSpace}{0ex} % One blank lines % University guideline 
\setlength{\UofUnegDoubleSpace}{-2ex} % Space between chapters and sections with no text
\setlength{\UofUnegSingleSpace}{-1ex} % Space between sections and subsections with no text
% 11 pt font is 13.6 pt

%----------------------------------------------------------------------%
%              Text/Figure/Table/Equation spacing
%----------------------------------------------------------------------%

\RequirePackage{setspace} % Single/Doublespacing options
\RequirePackage{etoolbox}

% Add University of Utah head space above and below figures/tables/equations
%\AtBeginEnvironment{figure}{}    %{\addvspace{\UofUHeadSpace}}
%\AtEndEnvironment{figure}{}    %{\addvspace{\UofUHeadSpace}}
%\AtBeginEnvironment{table}{}    %{\addvspace{\UofUHeadSpace}}
%\AtEndEnvironment{table}{}    %{\addvspace{\UofUHeadSpace}}
%\AtBeginEnvironment{equation}{}    %{\addvspace{\UofUHeadSpace}}
%\AtEndEnvironment{equation}{}    %{\addvspace{\UofUHeadSpace}}
\AtBeginEnvironment{tabular}{\singlespacing}% Single spacing in tabular environment

%----------------------------------------------------------------------%
%                     Additional Packages
%----------------------------------------------------------------------%

\RequirePackage{subfiles} % Allows for individual components to be created

\RequirePackage{lipsum} % random paragraphs # 66 & 75 are the shortest (4 lines each)

% Required for tables that span multiple pages (used in the symbols, 
% abbreviations and physical constants pages)
\RequirePackage{longtable} 
\RequirePackage{booktabs} % Required for better table rules (\toprule/\midrule/\bottomrule)

%----------------------------------------------------------------------%
%    Graphicsx package with option to turn on/off images for fast compiling
%----------------------------------------------------------------------%
%\RequirePackage{etoolbox} % When using ubuntu, this needs to be before tabular
\newbool{graphicDraft} % Option to turn graphics on/off
\DeclareOption{graphicFinal}{\boolfalse{graphicDraft}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\ProcessOptions\relax
%\RequirePackage[draft]{graphicx}
\ifbool{graphicDraft}
    {\RequirePackage[draft]{graphicx}}
    {\RequirePackage[final]{graphicx}}

\RequirePackage{subcaption} % package to create subfigure 
\RequirePackage{animate} % Animating stacks of images (pdf, png, etc.)
\RequirePackage{pdfpages} % Attach PDFs in the document

%----------------------------------------------------------------------%
%                   single spacing for all captions
%----------------------------------------------------------------------%

\RequirePackage[font=singlespacing]{caption}

%----------------------------------------------------------------------%
%                           float package
%----------------------------------------------------------------------%

% Allows figures and tables to be set in place "[H]"
\RequirePackage{float}

%----------------------------------------------------------------------%
%                           TodoNotes package
%----------------------------------------------------------------------%

\RequirePackage{todonotes} % to do notes

%----------------------------------------------------------------------%
%                       Geometry of the document
%----------------------------------------------------------------------%
\RequirePackage{geometry}
\geometry{
    paper          = letterpaper, % Change to letterpaper for US letter
    inner          = 1.25in, % Inner margin
    outer          = 1.25in, % Outer margin
    bindingoffset  = 0in,    % Binding offset
    top            = 1in,    % Top margin
    bottom         = 1in,    % Bottom margin
    %head           = 2in,   % header height % University guideline
    %showframe,              % Uncomment to show how the type block is set on the page
}


%----------------------------------------------------------------------%
%                               COLORS
%----------------------------------------------------------------------%

\RequirePackage{xcolor} % Required for specifying custom colors
\definecolor{BLACK}{HTML}{000000}
\definecolor{UOFURED}{HTML}{CC0000}
\definecolor{midnightBlue}{HTML}{145680} % midnight blue
\definecolor{darkBrown}{HTML}{654321} % Dark brown
\definecolor{arsenic}{HTML}{3B444B}
\definecolor{darkGreen}{HTML}{00b300} % Dark green
\definecolor{tab:blue}{HTML}{1f77b4} % python

%----------------------------------------------------------------------%
%                       Chapter heading font size
%----------------------------------------------------------------------%

\RequirePackage{sectsty}

\chapternumberfont{\normalsize} 
\chaptertitlefont{\normalsize}


%----------------------------------------------------------------------%
%                            Title spacing
%----------------------------------------------------------------------%

\RequirePackage[nostruts,compact]{titlesec} % Adjust title headers % option with no struts, compact

% Force Heading space after chapter/section/subsection/subsubsections to be the same
%    \titlespacing*{<command>}
%                  {<left>}
%                  {<before-sep>}
%                  {<after-sep>}
%                  [<right-sep>]

\titlespacing{\chapter}
             {0in}
             {\UofUOneInChapterSpace}
             {\UofUHeadSpace}
             [0in]

\titlespacing{\section}
             {0in}
             {\UofUHeadSpace}
             {\UofUDoubleSpace}
             [0in]

\titlespacing{\subsection}
             {0in}
             {\UofUHeadSpace}
             {\UofUDoubleSpace} % 0.5\UofUDoubleSpace
             [0in]

\titlespacing{\subsubsection}
             {0in}
             {\UofUHeadSpace}
             {0in} % 0.5\UofUDoubleSpace
             [0in]

\titlespacing{\paragraph}
             {1em}
             {-0.5\baselineskip}
             {\UofUDoubleSpace}
             [0in]

\titlespacing{\subparagraph}
             {1em}
             {-0.5\baselineskip}
             {\UofUDoubleSpace}
             [0in]

\titlespacing{\appendixchapter}
             {0in}
             {\UofUOneInChapterSpace}
             {\UofUHeadSpace}
             [0in]

% Double spaced chapter titles
% Single spaced headers (section/subsection/subsubsections/paragraph)
\titleformat{\chapter}  % command
    [display]           % shape (Creates new line after the word "CHAPTER ii"
    {\color{UOFURED}\centering\normalfont\Large\bfseries\doublespacing} % format
    {\MakeUppercase{\chaptertitlename} \vspace{\UofUHeadSpace} \Large\thechapter} % label
    {0in}               % separations (horizontal)
    {\MakeUppercase}    % before
    [\vspace{\UofUHeadSpace}]                  % After (leave blank)

\titleformat{\section}  % command
    [hang]           % shape %
    {\centering\singlespacing\normalfont\Large\bfseries} % format
    {\thesection} % label
    {1ex}               % separations (horizontal)
    {}                  % before
    [\vspace{\UofUDoubleSpace}]                  % After (leave blank)

\titleformat{\subsection}  % command
    [hang]              % shape
    {\centering\singlespacing\normalfont\large\bfseries} % format
    {\thesubsection} % label
    {1ex}               % separations (horizontal)
    {}                  % before
    [\vspace{\UofUDoubleSpace}]                  % After (leave blank)

\titleformat{\subsubsection}  % command
    [hang]                 % shape
    {\centering\singlespacing\normalfont\normalsize\bfseries} % format
    {\thesubsubsection} % label
    {1ex}               % separations (horizontal)
    {}                  % before
    [\vspace{\UofUDoubleSpace}]                  % After (leave blank)

\titleformat{\paragraph}  % command
    [runin]                 % shape
    {\centering\singlespacing\normalfont\normalsize\bfseries} % format
    {\theparagraph} % label
    {1ex}               % separations (horizontal)
    {}                  % before
    [\hspace*{1ex}]                  % After (leave blank)

\titleformat{\subparagraph}  % command
    [runin]                 % shape
    {\centering\singlespacing\normalfont\normalsize\bfseries} % format
    {\thesubparagraph} % label
    {1ex}               % separations (horizontal)
    {}                  % before
    [\hspace*{1ex}]                  % After (leave blank)


%----------------------------------------------------------------------%
%                            Paragraph spacing
%----------------------------------------------------------------------%

% Force indent paragraphs and skip space between paragraphs
%\RequirePackage[indent,skip=\UofUDoubleSpace]{parskip}
%\setlength{\parskip}{1em}

% Trying to prevent space between paragraphs from changing (5/4/2021)
%\setlength{\parindent}{4em}
%\setlength{\parskip}{1em}

%----------------------------------------------------------------------%
%          Figures on following page after being referenced
%----------------------------------------------------------------------%

% Call this to force an image on the next page if need be.
\RequirePackage{afterpage}


%----------------------------------------------------------------------%
%                               References
%----------------------------------------------------------------------%
% `final' to prevent disabling and reference links to be split 
% up on different lines
\RequirePackage[final]{microtype}  


\RequirePackage[defernums=true,
                hyperref,
                backref,
                backend=biber,
                style=ieee,
                natbib=true,
                refsegment=chapter,
                block=space
                ]{biblatex} % Use the bibtex backend, biber

\setlength\bibitemsep{1.5\itemsep} % Spacing between items in bilbiography

% Define typesetting of references heading
\defbibheading{references}[References]{% 
    \section{#1}% Add section so that the table of contents adds the chapter references
    \markboth{#1}{#1}%
    \singlespacing
}

%----------------------------------------------------------------------%
%                           Final Bibliography
%    Add type=book, article, online, thesis for specific references
%----------------------------------------------------------------------%

\def\finalBibliography{
    {\singlespacing
    \printbibliography[heading=bibintoc, title={\uppercase{Bibliography}}] % Whole
    }
}
    
%----------------------------------------------------------------------%
%                             Hyperlinks
%----------------------------------------------------------------------%
\RequirePackage{hyperref}
\hypersetup{pdfpagemode={UseOutlines},
    bookmarksopen=true,
    bookmarksopenlevel=0,
    hypertexnames=true, % reference roman numerals and numeric pages
    colorlinks=true,% Set to false to disable coloring links
    citecolor=darkBrown,% The color of citations
    linktoc=all,
    linkcolor=UOFURED,% The color of references to document elements (sections, figures, etc)
    urlcolor=midnightBlue,% The color of hyperlinks (URLs) 
    pdfstartview={FitV},
    unicode,
    breaklinks=true, % split up links
}

\RequirePackage{doi} % DOI field in references
\RequirePackage{url} % URL allowed to split up

\RequirePackage[anythingbreaks]{breakurl} % breaks URLs with the "\burl{xxxx@xxxxx.com}"  (After hyperref)

% Use capitalization of all references 
\RequirePackage[capitalise,noabbrev,nameinlink]{cleveref} % Needs to be loaded after hyperref
\crefname{paragraph}{Paragraph}{Paragraphs}% Allow for paragraphs to be referenced
\crefname{subparagraph}{Sub Paragraph}{Sub Paragraphs}% Allow for sub paragraphs to be referenced

% Remove Fig from subcaption references
\captionsetup[sub]{labelsep=none, subrefformat=parens} % Parenthesis around subfigure references
%\crefrangelabelformat{subfigure}{(#3\crefstripprefix{#2}{#1}#4--#5\crefstripprefix{#1}{#2}#6)}

%----------------------------------------------------------------------%
%  Adjust table of contents, list of figures, list of tables specifically
%----------------------------------------------------------------------%

\RequirePackage[titles]{tocloft} % Treat the headers as titles
\renewcommand{\cftchapdotsep}{\cftdotsep}   % Puts dots after chapter entries
\renewcommand{\contentsname}{\color{UOFURED}\hfill \Large CONTENTS \hfill}  % Normal font
\renewcommand{\listfigurename}{\color{UOFURED}\hfill \Large LIST OF FIGURES \hfill}  % Normal font
\renewcommand{\listtablename}{\color{UOFURED}\hfill \Large LIST OF TABLES \hfill}  % Normal font

% TOC subsection indents
\setlength{\cftsecindent}{1em}
\setlength{\cftsubsecindent}{2em}
\setlength{\cftsubsubsecindent}{3em}
\setlength{\cftparaindent}{4em}
\setlength{\cftsubparaindent}{5em}

%\setlength{\cftsecnumwidth}{3em}
%\setlength{\cftsubsecnumwidth}{3em}
%\setlength{\cftsubsubsecnumwidth}{3em}
%\setlength{\cftparanumwidth}{3em}
%\setlength{\cftsubparanumwidth}{3em}

\RequirePackage{xfp} % Evaluate in
% Define some awesome transcendental numbers, because why not?  :)
% To make things more exciting, we'll use Ramanujan's infinite series (20
% terms) to approximate pi
% 1/pi = (2*sqrt(2))/9801*sum_{k=0}^{\infty}((4*k)!*(1103 + 26390*k))/((k!)^4*(396^{4*k}))
\newcommand{\piRamanujan}{\fpeval{1/(2*sqrt(2)/9801*(1103.0 +
2.6831974348925308e-05 + 2.2453850201136644e-13 + 1.995074994495897e-21 +
1.8393545314677564e-29 + 1.7358835411230251e-37 + 1.6645493651891015e-45 +
1.6146422359450685e-53 + 1.579923842975531e-61 + 1.5565046601683694e-69 +
1.5418292843540324e-77 + 1.5341485332197387e-85 + 1.5322254371957438e-93 +
1.5351621561841798e-101 + 1.5422932991985167e-109 + 1.5531175881810274e-117 +
1.5672526255646388e-125 + 1.5844040989143966e-133 + 1.604344295911354e-141 +
1.6268967912330585e-149))}}

% Define functions
\RequirePackage{xparse}
%\ExplSyntaxOn
%\NewDocumentCommand{\myMathFunction}{m}
%    { \fp_eval:n {(fact(4*(#1))*(1103 + 26390*(#1)))/(fact(#1)^4*396^(4*(#1)))}
%\ExplSyntaxOff

\newcommand{\piLeibniz}{\fpeval{4*(1 - 1/3 + 1/5 - 1/7 + 1/9 - 1/11 + 1/13)}}
\newcommand{\goldenRatio}{\fpeval{(1 + sqrt(5))/2}}
\newcommand{\piIntegral}{\fpeval{\int_{-1}^{1}\frac{1}{sqrt{1 - x^2}}dx}} % Checking error message

% Dot separation definition
\renewcommand\cftchapdotsep{\piRamanujan}
\renewcommand\cftsecdotsep{\piRamanujan}
\renewcommand\cftsubsecdotsep{\piRamanujan}
\renewcommand\cftsubsubsecdotsep{\piRamanujan}
\renewcommand\cftparadotsep{\piRamanujan}
\renewcommand\cftsubparadotsep{\piRamanujan}

\renewcommand\cftchapafterpnum{\vskip2pt} % Spacing of the TOC items
\renewcommand\cftsecafterpnum{\vskip2pt} % Spacing of the TOC items
\renewcommand\cftsubsecafterpnum{\vskip2pt} % Spacing of the TOC items

\setlength{\cfttabindent}{0em} % Force LOT to be the correct width
\setlength{\cftfigindent}{0em} % Force LOF to be the correct width


%----------------------------------------------------------------------%
%                           Algorithm writing
%----------------------------------------------------------------------%

\RequirePackage{algpseudocode}
\RequirePackage{algorithm}


%----------------------------------------------------------------------%
%                          Thesis Title Page
%----------------------------------------------------------------------%
\def\thesistype#1{\gdef\@thesistype{#1}} % thesis or dissertation
\def\degreeAchieved#1{\gdef\@degreeAchieved{#1}} % exact degree being gotten
\def\department#1{\gdef\@department{#1}} % Department
\def\departmentLink#1{\gdef\@departmentLink{#1}} % Department
\def\submitdate#1{\gdef\@submitdate{#1}} % Submission date.
\newcommand\skiplines[1]{\vspace{#1\baselineskip}} % function for skipping n number of lines

\newcommand\thesisTitle{
    {
    %\newpage
    \thispagestyle{empty}% Doesn't count in table of contents
    \begin{minipage}[c]{\textwidth}
    %\chapter*{\color{BLACK}\uppercase{\@title}} % This doesn't work
        \vspace{1in}
        \centering
        \begin{spacing}{2} % university requirement
            {\doublespacing\Large\bfseries\MakeUppercase{\@title}}\\
        \end{spacing}
        \large
        \skiplines{4} % university requirement
        \begin{spacing}{2} % university requirement
            by \\
            \@author \\
        \end{spacing}
        \skiplines{4} % university requirement
        A \@thesistype ~submitted to the faculty of\\
        The University of Utah\\
        in partial fulfillment of the requirements for the degree of \\
        \skiplines{4} % university requirement
        {\@degreeAchieved}\\
        \skiplines{4} % university requirement
        \begin{spacing}{2} % university requirement
            \href{\@departmentLink}{\@department}\\
            \href{http://utah.edu}{The University of Utah}\\ % 
            \@submitdate
        \end{spacing}
    \end{minipage}
    \clearpage
    }
}


%----------------------------------------------------------------------%
%                          Copyright page
%----------------------------------------------------------------------%
\def\copyrightyear#1{\gdef\@copyrightyear{#1}} % Copyright year

\newcommand\copyrightpage{
    {\newpage
    \thispagestyle{empty}% Doesn't count in table of contents
    %\phantomsection
    %\pdfbookmark{Copyright year}{toc} % Add bookmark to PDF
    \topskip0pt
    \vspace*{\fill}
    \noindent
    \begin{minipage}[c]{\textwidth}%
        \centering
        Copyright \copyright\ \@author\ \@copyrightyear \\
        \skiplines{1} % university requirement
        All Rights Reserved
    \end{minipage}%
    \vspace*{\fill}
    \newpage% Added 1995 by GBG; see Eric Eide notes.
    }
}

%----------------------------------------------------------------------%
%    The committee approval page is highly constrained.  Its format is not
%    exactly identical to the preprinted ones.
%----------------------------------------------------------------------%

% Define committee members
\def\committeeChair#1{\gdef\@committeeChair{#1}} % Committee chair #1
\def\advisorTitle#1{\gdef\@\advisorTitle{#1}} % Advisor title (not really used)
\def\committeeMemberII#1{\gdef\@committeeMemberII{#1}} % Committee chair #2
\def\committeeMemberIII#1{\gdef\@committeeMemberIII{#1}} % Committee chair #3
\def\committeeMemberIIII#1{\gdef\@committeeMemberIIII{#1}} % Committee chair #4
\def\committeeMemberIIIII#1{\gdef\@committeeMemberIIIII{#1}} % Committee chair #5

% Define Graduate Dean & Department Chair
\def\graduateDean#1{\gdef\@graduateDean{#1}} % Graduate Dean
\def\departmentChair#1{\gdef\@departmentChair{#1}} % Department Chair
\def\departmentChairTitle#1{\gdef\@departmentChairTitle{#1}} % Department Chair Title
\def\deptmentCollegeSchool#1{\gdef\@deptmentCollegeSchool{#1}} % Department/College/School

% Define approval dates
\def\approvaldepartment#1{\gdef\@approvaldepartment{#1}}
\def\chairDateApproved#1{\gdef\@chairDateApproved{#1}}
\def\committeeMemberIIDateApproved#1{\gdef\@committeeMemberIIDateApproved{#1}}
\def\committeeMemberIIIDateApproved#1{\gdef\@committeeMemberIIIDateApproved{#1}}
\def\committeeMemberIIIIDateApproved#1{\gdef\@committeeMemberIIIIDateApproved{#1}}
\def\committeeMemberIIIIIDateApproved#1{\gdef\@committeeMemberIIIIIDateApproved{#1}}

\def\bfunderline#1{\underline{\kern 1in \textbf{#1}\kern 1in}}

\RequirePackage{tabularx} % pagewidth tables
\RequirePackage{array} % Specify width of tables
\RequirePackage{ragged2e} % Allows for justification of sections

\long \def \dissertationapproval{%
    \newpage
    %
    \thispagestyle{empty}% Doesn't count in table of contents
    %\phantomsection
    %\pdfbookmark{Dissertation Approval}{toc} % Add bookmark to PDF
    \vspace*{\fill}
    \noindent
    \begin{minipage}[c]{\textwidth}
        \centering
        \textbf{\Large The University of Utah Graduate School} \\
        \skiplines{2} % university requirement
        \textbf{\Large STATEMENT OF \ifthenelse{\equal{\@thesistype}{thesis}}{THESIS}{DISSERTATION}\ APPROVAL}
        \skiplines{2} % university requirement
        %
        \justify
        \begin{doublespace}
            The \@thesistype\ of\hspace{1in}\bfunderline{\@author} \\
            \skiplines{2} % university requirement
            has been approved by the following supervisory committee members:

            \begin{table}[H] % [htbp]
                \centering    \begin{tabular}{>{\centering\arraybackslash}p{0.6\textwidth}p{0.1\textwidth}>{\centering\arraybackslash}p{0.2\textwidth}}
                    \bfseries{\@committeeChair} & \multicolumn{1}{l}{ , Chair} & \multicolumn{1}{c}{\bfseries{\@chairDateApproved}} \\
                    \cmidrule{1-1}\cmidrule{3-3}          &       & \tiny{Date Approved} \\
                        &    &    \\ % add extra row between the committee members
                    \bfseries{\@committeeMemberII} & \multicolumn{1}{l}{ , Member} & \multicolumn{1}{c}{\bfseries{\@committeeMemberIIDateApproved}} \\
                    \cmidrule{1-1}\cmidrule{3-3}          &       & \tiny{Date Approved} \\
                        &    &    \\ % add extra row between the committee members
                    \bfseries{\@committeeMemberIII} & \multicolumn{1}{l}{ , Member} & \multicolumn{1}{c}{\bfseries{\@committeeMemberIIIDateApproved}} \\
                    \cmidrule{1-1}\cmidrule{3-3}          &       & \tiny{Date Approved} \\
                        &    &    \\ % add extra row between the committee members
                    \ifthenelse{\equal{\@thesistype}{thesis}}{}{
                    \bfseries{\@committeeMemberIIII} & \multicolumn{1}{l}{ , Member} & \multicolumn{1}{c}{\bfseries{\@committeeMemberIIIIDateApproved}} \\
                    \cmidrule{1-1}\cmidrule{3-3}          &       & \tiny{Date Approved} \\
                        &    &    \\ % add extra row between the committee members
                    \bfseries{\@committeeMemberIIIII} & \multicolumn{1}{l}{ , Member} & \multicolumn{1}{c}{\bfseries{\@committeeMemberIIIIIDateApproved}} \\
                    \cmidrule{1-1}\cmidrule{3-3}          &       & \tiny{Date Approved} \\
                    }
                \end{tabular}%
            \end{table}%
            
            \justify
            and by \hspace{1in}\bfunderline{\@departmentChair}, {\@departmentChairTitle} of \\
            the {\@deptmentCollegeSchool} of \hspace{1in}\bfunderline{\@approvaldepartment} \\
            and by \bfunderline{\@graduateDean}, Dean of The Graduate School.\\
        \end{doublespace}
    \end{minipage}
    \vspace*{\fill}
}

%----------------------------------------------------------------------%
%                              Abstract
%----------------------------------------------------------------------%
\def\abstractString#1{\gdef\@abstractString{#1}} % Abstract

\newcommand\abstractPage{
    \newpage                    % blank page to have the abstract start on page 3 (right side)
    \thispagestyle{empty}       % blank page to have the abstract start on page 3 (right side)
    {\newpage
    \chapter*{Abstract}
        % manually set to page iii (University requirement)
        \setcounter{page}{3}
        \addcontentsline{toc}{chapter}{\uppercase{Abstract}}
        \hspace{\parindent}
        \begin{doublespace}
            \@abstractString
        \end{doublespace}
    }
}


%----------------------------------------------------------------------%
%                           Dedication page
%----------------------------------------------------------------------%
\def\dedication#1{\gdef\@dedication{#1}} % Dedication

\newcommand\dedicationPage{
    {\newpage
        % Don't include in page count
        \clearpage\thispagestyle{empty}\addtocounter{page}{-1} 
        \topskip0pt
        \vspace*{\fill}
        \noindent
        \begin{minipage}[c]{.95\textwidth}%
            \centering
            \begin{doublespace}
                \@dedication
            \end{doublespace}
        \end{minipage}%
        \vspace*{\fill}
    }
}

%----------------------------------------------------------------------%
%                           Frontispiece page
%----------------------------------------------------------------------%
\def\frontispiece#1{\gdef\@frontispiece{#1}} % Frontispiece

\newcommand\frontispiecePage{
    {\newpage        
        % Don't display the page number but still add to the total count
        \clearpage\thispagestyle{empty}
        \topskip0pt
        \vspace*{\fill}
        \noindent
        \begin{minipage}[c]{\textwidth}%
            \centering
            \includegraphics[width=\textwidth]{{\@frontispiece}}
        \end{minipage}%
        \vspace*{\fill}
    }
}


%----------------------------------------------------------------------%
%                  Epigraph page (Inspirational quote)
%----------------------------------------------------------------------%
\RequirePackage{epigraph} % epigraph package
\def\epigraphQuote#1{\gdef\@epigraphQuote{#1}} % Epigraph quote
\def\epigraphAuthor#1{\gdef\@epigraphAuthor{#1}} % Epigraph author

%\setlength{\epigraphwidth}{0.6\textwidth} % Width of epigraph

\newcommand\epigraphPage{
    {\newpage
        % Don't display the page number but still add to the total count    
        \clearpage\thispagestyle{empty}
        \topskip0pt
        \vspace*{\fill}
        \noindent
        \begin{minipage}[c]{\textwidth}%
            \centering
            \epigraph{\Large{\@epigraphQuote}}{---{\@epigraphAuthor}}
        \end{minipage}%
        \vspace*{\fill}
    }
}


%----------------------------------------------------------------------%
%                       Acknowledgements page
%----------------------------------------------------------------------%
\def\Acknowledgement#1{\gdef\@Acknowledgement{#1}} % Acknowledgement

\newcommand\AcknowledgementPage{
    {
    \chapter*{Acknowledgements}
    \addcontentsline{toc}{chapter}{\uppercase{Acknowledgements}}
    \hspace{\parindent}
    \begin{doublespace}
        \@Acknowledgement
    \end{doublespace}
    }
}


%----------------------------------------------------------------------%
%                      Notation and Symbols Page
%----------------------------------------------------------------------%
\def\NotationAndSymbols#1{\gdef\@NotationAndSymbols{#1}} % Notation and Symbols

\newcommand\NotationSymbolsPage{
    {
    \chapter*{Notation and Symbols}
        \addcontentsline{toc}{chapter}{\uppercase{Notation and Symbols}}
        \begin{center}
            \@NotationAndSymbols
        \end{center}
    }
}


%----------------------------------------------------------------------%
%              Specify frontmatter, mainmatter, and backmatter
%----------------------------------------------------------------------%

\RequirePackage{fancyhdr}

\newcommand\frontMatter{%
    \hypersetup{bookmarksdepth=5} % Display bookmarks in PDF
    
    \cleardoublepage
    \pagenumbering{roman}
    
    % Redefine the pagestyle plain because when \chapter{...} is evoked, 
    % it ignores the current page style
    \fancypagestyle{plain}{%
      \renewcommand{\headrulewidth}{0pt}%
      \fancyhf{} % University requirement that pages with chapter titles don't show numbers
      %\fancyhead[R]{\thepage} % Uncomment to put page numbers on chapter title page
    }
}

\newcommand\mainMatter{%
    
    \hypersetup{bookmarksdepth=5} % Display bookmarks in PDF
    
    % Add the word "Chapters" in the TOC as per University requirements
    \addtocontents{toc}{\vspace{1em}\noindent{\bf\uppercase{Chapters}}\hfill\par} 
    \cleardoublepage
    
    % Page numbering in the top-right hand corner
    \pagestyle{fancy}
    \fancyhf{} % clear the fancy header
    \lhead{} % nothing in the left corner
    \rhead{\thepage} % page number in the top right corner
    \renewcommand{\headrulewidth}{0pt} % hide the line at the top of the fancy header
    
    % Redefine the pagestyle plain because when \chapter{...} is evoked, 
    % it ignores the current page style
    \fancypagestyle{plain}{%
      \renewcommand{\headrulewidth}{0pt}%
      \fancyhf{} % University requirement that pages with chapter titles don't show numbers
      %\fancyhead[R]{\thepage} % Uncomment to put page numbers on chapter title page
    }
    
    \pagenumbering{arabic} % University requirement (numbered pages)
    \doublespacing % University requirement (Double spaced text)
}

%\RequirePackage[page,toc,titletoc,title]{appendix}

\newcommand\backMatter{%
    % Add the word "Appendices" in the TOC as per University requirements
    \addtocontents{toc}{\vspace{1em}\noindent{\bf\uppercase{Appendices}}\hfill\par} 
    %\cleardoublepage
    
    % Do not display sections/subsections/subsubsections/paragraphs/subparagraphs
    %\addtocontents{toc}{\setcounter{tocdepth}{-10}}
    
    \hypersetup{bookmarksdepth=5} % Display bookmarks in PDF
    
    % Only display Appendix titles in TOC
    \addtocontents{toc}{\protect\setcounter{tocdepth}{0}}
    
    % Create a new page for appendix chapters (Center the title on the page)
    \titlespacing{\chapter}
             {0in}
             {-\UofUOneInChapterSpace}
             {\UofUHeadSpace}
             [0in]
             
    \titleformat{\chapter}
        [display]
        {\color{UOFURED}\centering\normalfont\Large\bfseries\doublespacing} % format
        {\vspace*{\fill} \MakeUppercase{\chaptertitlename} \Large\thechapter} % label \vspace{\UofUHeadSpace} 
        {1ex}
        {\MakeUppercase}
        [\vspace*{\fill}\newpage]
    
    \appendix
}



\newcommand\postAppendix{%
    
    \makeatletter
    \@addtoreset{chapter}{part}
    \makeatother 
    
    \hypersetup{bookmarksdepth=5} % Display bookmarks in PDF
    
    % Only display Appendix titles in TOC
    \addtocontents{toc}{\protect\setcounter{tocdepth}{0}}
    
    % Page numbering in the top-right hand corner
    \pagestyle{fancy}
    \fancyhf{} % clear the fancy header
    \lhead{} % nothing in the left corner
    \rhead{\thepage} % page number in the top right corner
    \renewcommand{\headrulewidth}{0pt} % hide the line at the top of the fancy header
    
    % Redefine the pagestyle plain because when \chapter{...} is evoked, 
    % it ignores the current page style
    \fancypagestyle{plain}{%
      \renewcommand{\headrulewidth}{0pt}%
      \fancyhf{} % University requirement that pages with chapter titles don't show numbers
      %\fancyhead[R]{\thepage} % Uncomment to put page numbers on chapter title page
    }
    
    \pagenumbering{arabic} % University requirement (numbered pages)
    \doublespacing % University requirement (Double spaced text)
    
}


%----------------------------------------------------------------------%
%%                          Custom table of contents
%----------------------------------------------------------------------%
\renewcommand\cftfigafterpnum{\vskip5pt\par} % list of figure spacing
\renewcommand\cfttabafterpnum{\vskip5pt\par} % list of table spacing

\newcommand\tableOfContents{
    {
    \hypersetup{linkcolor=blue} % Change color if you want something different
    \clearpage
    \pdfbookmark{Table of Contents}{toc} % Add bookmark to PDF
    \tableofcontents % Prints the main table of contents
    
    \clearpage
    \phantomsection
    \addcontentsline{toc}{chapter}{\uppercase{List of Figures}} % Adds to TOC
    \listoffigures % Prints the list of figures    
    
    \clearpage
    \phantomsection
    \addcontentsline{toc}{chapter}{\uppercase{List of Tables}} % Adds to TOC
    \listoftables % Prints the list of tables
    }
}

% turn off the error for \uppercase in chapter titles
\pdfstringdefDisableCommands{\let\uppercase\relax} 

%----------------------------------------------------------------------%
%%            % Mendeley bib file URL issue fix
%----------------------------------------------------------------------%

\DeclareSourcemap{
    \maps{
        \map{ % Replaces '{\_}', '{_}' or '\_' with just '_'
            \step[fieldsource=url,
                  match=\regexp{\{\\\_\}|\{\_\}|\\\_},
                  replace=\regexp{\_}]
        }
        \map{ % Replaces '{'$\sim$'}', '$\sim$' or '{~}' with just '~'
            \step[fieldsource=url,
                  match=\regexp{\{\$\\sim\$\}|\{\~\}|\$\\sim\$},
                  replace=\regexp{\~}]
        }
        \map{ % Replaces '{\$}'
            \step[fieldsource=url,
                  match=\regexp{\{\\\x{26}\}},
                  replace=\regexp{\x{26}}]
        }
    }
}

%----------------------------------------------------------------------%
%  % Prevent Orphans and Widows (lines begining at the bottom/top of a page)
%----------------------------------------------------------------------%

\RequirePackage[defaultlines=2,all]{nowidow}

%----------------------------------------------------------------------%
%%                  % Specific chemistry formulations
%----------------------------------------------------------------------%

\RequirePackage{chemfig}

%----------------------------------------------------------------------%
%          Figures and tables at the end of the chapter - I can't seem to get this to work to switch
%----------------------------------------------------------------------%

% Call this to force all figures/tables to be at the end of the chapter
\RequirePackage[disable]{endfloat} % figlist,tablist

%----------------------------------------------------------------------%
%%                  % if then else statements
%----------------------------------------------------------------------%
\RequirePackage{ifthen}

%----------------------------------------------------------------------%
%%                  % LaTeX symbol
%----------------------------------------------------------------------%
\RequirePackage{xspace}
\makeatletter % changes the catcode of @ to 11
\newcommand{\latexSymbol}{\LaTeX\ \xspace}
\makeatother % changes the catcode of @ back to 12

%----------------------------------------------------------------------%
%%                  % LaTeX symbol
%----------------------------------------------------------------------%
\RequirePackage{hologo}

\endinput
% Christopher Creveling
