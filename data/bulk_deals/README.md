# NSE bulk deals

Drop NSE bulk-deal CSV exports in this directory; every `*.csv` here is loaded and
concatenated, so additional periods can be added without changing any code.

Source: NSE India, "Bulk Deals" report (Market Data → Large Deals). A bulk deal is a
transaction in which a single client buys or sells more than 0.5% of a company's listed
equity shares in one session. The exchange publishes the list **after the close of the
trading session**, which is why the features built from it are treated as available at the
close of the same session.

Expected columns (header whitespace and the UTF-8 BOM are handled by the loader):

```
Date, Symbol, Security Name, Client Name, Buy / Sell, Quantity Traded,
Trade Price / Wght. Avg. Price, Remarks
```

Quantities use Indian digit grouping (`3,90,000`), which the loader parses.

Note that large-capitalisation stocks almost never appear here: moving 0.5% of Reliance or
TCS in a single deal is impractical. The dataset is dominated by small and mid-cap symbols,
and the application reports per-ticker coverage before using these features.
