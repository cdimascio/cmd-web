---
title: Dotenv for Java
date: "2023-03-01T19:15:00.121Z"
template: "post"
draft: false
slug: "dotenv-for-java"
category: "Java"
tags:
  - "dotenv"
  - "java"
description: "This article introduces dotenv-java, a library designed to simplify environment variable management for Java applications. Originating from the Ruby community and adhering to the twelve-factor app methodology, dotenv provides a consistent, language-agnostic approach to configuration across development, testing, staging, and production environments. By loading variables from a simple .env file, it emulates deployment environments while keeping local development straightforward. Dotenv avoids overriding host environment variables and supports various languages, including Java, Python, Go, and Node.js. The article includes examples of using dotenv-java, highlighting its utility for managing configuration effectively and consistently across environments."
---

Developers often grapple with resource configuration, managing indpendent settings for environments such as dev, staging, production, and local development. There are numerous libraries and frameworks that employ vastly different configuration mechanism, sometimes relying on property files, YAML, JSON, INI, TOML, and more. Fortunately, a common mechanism exists that supported pervasevly across most platforms, environment variables.

Enter dotenv-java, a widely adopted solution originating from the Ruby community. dotenv provides a consistent and straightforward method for configuring applications using environment variables, aligning with the principles of the [twelve-factor methodology](https://12factor.net/).

A dotenv library exists for most prominent languages including Javascript, Go, Rust, Python, and of course Ruby. Dotenv simplifies the process by loading variables by using simple .env file at startup. the contents of the .env file are simply key value pairs, where the key represnts the name of the environment variable name and the value, its value. The key, value pair in a .env file will never override the actual environment. It’s main purpose is to simplify development, seamlessly simulating a deployment environmemt, but using settings that make sense of local development.

By using environment variables, a robust widely adopted mechanis for managing configuration, Dotenv ensures a uniform configuration experience across development, testing, staging, and production environments, achieved by populating a .env (typical for local development) or leveraging environment variables (typical of deployed environments).

Dotenv describes itself as such (from bkeepers original Ruby dotenv)

__Storing configuration in the environment is one of the tenets of a [twelve-factor app]((https://12factor.net/)). Anything that is likely to change between deployment environments–such as resource handles for databases or credentials for external services–should be extracted from the code into environment variables.

But it is not always practical to set environment variables on development machines or continuous integration servers where multiple projects are run. dotenv loads variables from a .env file into ENV when the environment is bootstrapped.__

Here a brief list of dotenv implementations for common language:

- Node.js [https://github.com/motdotla/dotenv](https://github.com/motdotla/dotenv)
- Go [https://github.com/joho/godotenv](https://github.com/joho/godotenv)
- Python [https://github.com/theskumar/python-dotenv](https://github.com/theskumar/python-dotenv)
- Ruby (the trailblazer) [https://github.com/bkeepers/dotenv](https://github.com/bkeepers/dotenv)
- Java [https://github.com/cdimascio/dotenv-java](https://github.com/cdimascio/dotenv-java)
- Kotlin [https://github.com/cdimascio/dotenv-kotlin](https://github.com/cdimascio/dotenv-kotlin)
- Rust [https://crates.io/crates/dotenv](https://crates.io/crates/dotenv)

his article focuses on the dotenv-kotlin

Getting Started with java-dotenv: Create a .env file in your project’s root:

Use `dotenv.get("...")` instead of Java’s `System.getenv(...)`. Here’s why.

Create a `.env` file in the root of your project

```shell
# formatted as key=value
MY_ENV_VAR1=some_value
MY_EVV_VAR2=some_value #some value comment
```

```java
Dotenv dotenv = Dotenv.load();
dotenv.get("MY_ENV_VAR1")
```

Iterate over environment variables
Note, environment variables specified in the host environment take precedence over those in .env.

```java
for (DotenvEntry e : dotenv.entries()) {
    System.out.println(e.getKey());
    System.out.println(e.getValue());
}
```

For advanced features, see java-dotenv. If you like what you see, star it on github

Happy coding!