Feature: Common utilities
  In an automated pipeline
  I want to re-use some common functions

  Scenario: URI paths from labels
    Given a table of labels and corresponding paths
      | label                                      | path                                    |
      | Something with (brackets at the end)       | something-with-brackets-at-the-end      |
      | Family: Partner (for immediate settlement) | family-partner-for-immediate-settlement |
      | Aruba and Curaçao                          | aruba-and-curacao                       |
    Then the pathify function should convert labels to paths