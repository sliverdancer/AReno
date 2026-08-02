# Tau3 source identity correction

This correction occurred during source acquisition, before installation,
environment qualification, model access, inference, training, or results.

`git ls-remote --tags` returns two identities for annotated tag `v1.0.1`:

- `b711c1ead46f55111bf765cf44d5da8bacc2d28c` for `refs/tags/v1.0.1`;
- `fc0055dc4e0a316c3f83133267fbd6faaa770992` for
  `refs/tags/v1.0.1^{}`.

The former is the signed/annotated tag object; the latter is the source commit.
The initial lock incorrectly used the tag object as `commit`. `SOURCE_LOCK.json`
now stores both and `validate_checkout.py` verifies that the frozen tag object
peels to the frozen source commit. No scientific threshold, split, treatment,
or outcome was changed.
